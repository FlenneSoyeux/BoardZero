"""Module used to train the model, base on self-play"""

"""MODULE MEMORY"""
module MemoryModule

using ..Game, ..NNet
using Random, Flux
using DataStructures: Deque
using Serialization

include("../../params.jl")

export MemoryBuffer, save_to_file, load_from_file!, make_data

mutable struct Element
    input::Any    #NxNx2 , [:,:,1] pour playerToPlay et [:,:,2] pour l'autre
    playerToPlay::Int
    value_MCTS::Float32    # donné par le MCTS
    value_final::Float32   # 1 si victoire, 0.5 si draw, 0 si loss
    points_MCTS::Vector{Float32}    # delta points given by MCTS
    points_final::Vector{Float32}   # delta points at the end of the game
    pi_MCTS::Vector{Float32}         # N^2 donné par le MCTS
    surpriseWeight::Float32

    Element(input, playerToPlay, value_MCTS, value_final, points_MCTS, points_final, pi_MCTS, surpriseWeight) = new(copy(input), playerToPlay, value_MCTS, value_final, copy(points_MCTS), copy(points_final), copy(pi_MCTS), surpriseWeight)
    function Element(G::Game.AbstractGame, pi_MCTS::Vector{Float32}, value_MCTS::Float32, surpriseWeight::Float32, rng = Random.default_rng())
        input = Game.get_input_for_nn!(G, pi_MCTS, rng)
        return new(input, G.playerToPlay, value_MCTS, -1, zeros(Float32, LENGTH_POINTS), zeros(Float32, LENGTH_POINTS), pi_MCTS, surpriseWeight)
    end
    function Element(G::Game.AbstractGame, pi_MCTS::Vector{Float32}, value_MCTS::Float32, points_MCTS::Vector{Float32}, surpriseWeight::Float32, rng = Random.default_rng())
        input = Game.get_input_for_nn!(G, pi_MCTS, rng)
        points_final = -INF*ones(Float32, LENGTH_POINTS)
        if (try TYPE_GAME <: Azul catch; false end) 
            points_final[1] = G.scores[G.playerToPlay] - G.scores[3-G.playerToPlay]
        end
        return new(input, G.playerToPlay, value_MCTS, -1, copy(points_MCTS), points_final, pi_MCTS, surpriseWeight)
    end
end

const SIZE_REPLAYBUFFER = LEARNING_PARAMS["SIZE_REPLAYBUFFER"]
mutable struct MemoryBuffer
    replayBuffer::Deque{Element}
    MemoryBuffer() = new( Deque{Element}() )
end

function save!(M::MemoryBuffer, elem::Element)
    push!(M.replayBuffer, elem)
    if length(M.replayBuffer) > SIZE_REPLAYBUFFER
        popfirst!(M.replayBuffer)
    end
end

function save!(M::MemoryBuffer, G::Game.AbstractGame, pi_MCTS::Vector{Float32}, value_MCTS::Float32, points_MCTS::Vector{Float32}, surpriseWeight::Float32, rng = Random.default_rng())
    push!(M.replayBuffer, Element(G, pi_MCTS, value_MCTS, points_MCTS, surpriseWeight, rng))
    while length(M.replayBuffer) > SIZE_REPLAYBUFFER
        popfirst!(M.replayBuffer)
    end
end

function assign_winner(E::MemoryBuffer, winner, scores=0)
    G = new_game()
    for elem in E.replayBuffer
        if (try TYPE_GAME <: Azul catch; false end)
            # scores == [delta raw score end of game , delta rows, delta cols, delta colors]
            # elem.pointsfinal = [delta score end of round, delta score with other rounds, delta rows, delta cols, delta colors]

            elem.points_final[2] = scores[elem.playerToPlay][1] - scores[3-elem.playerToPlay][1] - elem.points_final[2]
            elem.points_final[[3,4,5]] = scores[elem.playerToPlay][ [2,3,4] ] - scores[3-elem.playerToPlay][ [2,3,4] ]

        elseif TYPE_GAME <: AbstractPointsGame
            elem.points_final = scores[elem.playerToPlay] - scores[3-elem.playerToPlay]
        end

        if winner == 0
            elem.value_final = 0.5
        elseif winner == elem.playerToPlay
            elem.value_final = 1
        else
            elem.value_final = 0
        end

    end
end

function assign_round(E::MemoryBuffer, scores)
    @assert (try TYPE_GAME <: Azul catch; false end)
    for elem in E.replayBuffer 
        if elem.points_final[end] == -INF
            elem.points_final[ 2:5 ] *= 0
            # We want : Delta s.t. Delta score previous round (=elem.points_final[1]) + Delta = Delta score next round (=scores)
            # Therefore Delta = Deltascores - elem.points_final[1]
            elem.points_final[1] = scores[elem.playerToPlay] - scores[3-elem.playerToPlay] - elem.points_final[1]

            # Also prepare terrain : in 2 we have Delta Score next round 
            elem.points_final[2] = scores[elem.playerToPlay] - scores[3-elem.playerToPlay]
        end 
    end
end

function save_to_file(E::MemoryBuffer)
    v = [[e.input, e.playerToPlay, e.value_MCTS, e.value_final, e.points_MCTS, e.points_final, e.pi_MCTS, e.surpriseWeight] for e in E.replayBuffer]
    println(length(v))
    serialize(DIRECTORY*"replayBuffer.jls", v)
end

function load_from_file!(M::MemoryBuffer; fileName="replayBuffer.jls")
    if !isfile(DIRECTORY*fileName)
        return false
    end
    println("Loading memory...")
    v = deserialize(DIRECTORY*fileName)
    for vec in v
        if length(vec) == 6
            println(vec)
            push!(M.replayBuffer, Element(vec[1], vec[2], vec[3], vec[4], vec[5], vec[5], vec[6], 0.089f0))  # s.t. e.points_MCTS = e.points_final
        elseif length(vec) == 7
            push!(M.replayBuffer, Element(vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], 0.089f0))  # s.t. e.points_MCTS = e.points_final
        else
            push!(M.replayBuffer, Element(vec...))
        end
        
        while length(M.replayBuffer) > LEARNING_PARAMS["SIZE_REPLAYBUFFER"]
            popfirst!(M.replayBuffer)
        end
    end
end

function make_data(memory::MemoryBuffer, nn::AbstractNN; range=nothing, surpriseWeight=false, shuffle=true)
    if surpriseWeight
        meanSurprise = sum([E.surpriseWeight for E in memory.replayBuffer]) / length(memory.replayBuffer)
        ff(x) = floor(x) + (rand() < x - floor(x))  # ff(3.8) = 3 or 4, 4 with 80% chance 
        f(s) = ff(0.5 + 0.5 * s / meanSurprise)     # in average, retrieves same length as original
        @assert isnothing(range)
        memoryIndexing = vcat([  [E for i in 1:f(E.surpriseWeight)]  for E in memory.replayBuffer]...)
    else
        memoryIndexing = isnothing(range) ? memory.replayBuffer : [E for E in memory.replayBuffer][range]
    end
    
    X = NNet.concatenate_inputs(nn, [E.input for E in memoryIndexing])
    target = hcat([ vcat(E.pi_MCTS, 
                            E.value_final+LEARNING_PARAMS["WEIGHT_vMCTS"]*(E.value_MCTS-E.value_final), 
                            E.points_final+LEARNING_PARAMS["WEIGHT_vMCTS"]*(E.points_MCTS-E.points_final)) 
                    for E in memoryIndexing ]...)
    if surpriseWeight
        println("Surprise : mean=", meanSurprise, " size memoryindex = ", length(memoryIndexing))
    end

    return Flux.DataLoader((X..., target), batchsize = LEARNING_PARAMS["BATCHSIZE"], shuffle=shuffle, partial=false)
end

end



"""MODULE TRAINER"""
module Trainer

using ..Algorithm, ..AZmodule, ..Game, ..NNet
using ..MemoryModule, ..Stats
using ..Arena, ..ParallelAZ
using Flux, AMDGPU, Random, Statistics, GPUArrays
using BSON: @save
using ParameterSchedulers
using Distributions: Categorical

include("../../params.jl")

export train_parallel

function train_parallel()
    ###use export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE" before running julia -t
    ##1) make N games in parallel and gather data
    ##2) train
    ##3) save / evaluate

    nn = are_all_nn_set() ? initialize_model(:last) : initialize_model(:last)

    NGAMES = LEARNING_PARAMS["NGAMES"]
    rngs::Vector{AbstractRNG} = [ Random.MersenneTwister( rand(Int) ) for i in 1:Threads.nthreads()]
    memory = MemoryBuffer()
    load_from_file!(memory)

    input =  NNet.get_input_shape(nn, NGAMES)
    #output = NNet.get_output_shape(nn, NGAMES)

    #STATS loading (ELO, games, etc...)
    all_stats = Stats.Stat[]
    for id in 1:10000
        S = Stats.load_stats(id)
        !isnothing(S) || break
        push!(all_stats, S)
    end
    Stats.print(all_stats)
    cur_stats = length(all_stats) > 0 ? Stats.Stat(all_stats[end]) : Stats.Stat()

    timeUntilSave = (cur_stats.id == 1) ? 0 : LEARNING_PARAMS["SAVE_EVERY"] #NN will be saved when timeUntilSave is 0

    opt_state = Flux.setup( Flux.AdamW( LEARNING_PARAMS["lr_max"], (0.9, 0.999), 0.025 ), nn)

    # START TRAINING
    for ep in 1:LEARNING_PARAMS["EPISODES"]
        println("\n-----------------------------\nEPISODE ", ep, " -- start")
        Flux.testmode!(nn)
        time_gpu::Float64 = 0

        Games::Vector{Game.AbstractGame} = [new_game() for i in 1:NGAMES]
        Games_after_eval::Vector{Game.AbstractGame} = [new_game() for i in 1:NGAMES]
        AZs::Vector{AZAlgorithm} = [AZAlgorithm(Games[i], nn) for i in 1:NGAMES]
        curs::Vector{AZmodule.Node} = [AZmodule.Node(Games[i]) for i in 1:NGAMES]
        PCR::Vector{Bool} = [ rand(rngs[1], [true, true, true, false]) for i in 1:NGAMES ]
        FP::Vector{Bool} = [ !PCR[i] && DO_FORCED_PLAYOUTS for i in 1:NGAMES ]
        ITER::Vector{Int} = [ PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"], PCR_REDUCTION) :  LEARNING_PARAMS["MCTS_ITER"] for i in 1:NGAMES ]

        memories = [MemoryBuffer() for i in 1:NGAMES]

        total_iter, real_iter = 0, 0
        sum_rounds = Threads.Atomic{Int}(0)
        time_run = time()   # en secondes
        cache = GPUArrays.AllocCache()
        time_before = 0
        time_after = 0

        total_time = @timed while !all([is_finished(G) for G in Games])
            #ITERATION DE AZ - before eval

            needNN = [true for i in 1:NGAMES]
            time_before -= time()
            Threads.@threads for i in 1:NGAMES
                if is_finished(Games[i]) || AZs[i].ROOT.N >= ITER[i]
                    needNN[i] = false
                    continue
                end

                curs[i] = ParallelAZ.before_nn_eval!(AZs[i], Games[i], Games_after_eval[i]; forced_playouts=FP[i])
                if !is_finished(Games_after_eval[i])
                    NNet.inplace_change!(nn, input, i, Game.get_input_for_nn(Games_after_eval[i]))
                end
                
            end
            time_before += time()

            #GPU EVALUATION
            time_gpu -= time()

            idx_not_finished = [i for i in 1:NGAMES if needNN[i]]

            if length(idx_not_finished) != 0
                GPUArrays.@cached cache begin
                    output = try 
                            NNet.predict(nn, input |> gpu) |> cpu
                        catch
                            println("----- error - retry-----")
                            NNet.predict(nn, input |> gpu) |> cpu
                        end
                end
                real_iter += length(idx_not_finished)
                total_iter += NGAMES
            end

            time_gpu += time()

            #ITERATION DE AZ - after eval
            time_after -= time()
            Threads.@threads for i in 1:NGAMES
                if is_finished(Games[i])
                    continue
                elseif needNN[i]
                    add_dirichlet = (!PCR[i] && length(AZs[i].ROOT.children) == 0)  # Normally happens only on first iteration
                    ParallelAZ.after_nn_eval!(curs[i], Games_after_eval[i], Games[i], output, i; rng=rngs[Threads.threadid()], add_dirichlet=add_dirichlet)   #just add dirichlet at root
                end

                prematureStop = false
                if PCR[i] && AZs[i].ROOT.N < ITER[i] && length(AZs[i].ROOT.children) >= 2
                    val = [c.N for c in AZs[i].ROOT.children]
                    partialsort!(val, 2 ; rev=true)
                    prematureStop = (val[1]-val[2]) > (ITER[i] - AZs[i].ROOT.N) 
                end

                if AZs[i].ROOT.N >= ITER[i] || prematureStop
                    temperature::Float32 = PCR[i] ? 0.0f0 : (TEMP_INIT - TEMP_FINAL) * BASE_HALFLIFE^Game.number_played_moves(Games[i]) + TEMP_FINAL
                    if length(AZs[i].ROOT.children) == 0
                        print(Games[i])
                        error("erreur !!")
                    end
                    move, pi_MCTS, v_MCTS, points_MCTS = AZmodule.get_move_temperature(AZs[i], Games[i], temperature; rng=rngs[Threads.threadid()], forced_playouts=FP[i])

                    surpriseWeight::Float32 = PCR[i] ? 0f0 : Flux.Losses.kldivergence( [c.pi_value for c in AZs[i].ROOT.children], [c.N/(AZs[i].ROOT.N-1) for c in AZs[i].ROOT.children])

                    keep_subtree!(AZs[i], move)

                    if !PCR[i]
                        MemoryModule.save!(memories[i], Games[i], pi_MCTS, v_MCTS, points_MCTS, surpriseWeight, rngs[Threads.threadid()])
                    end

                    play!(Games[i], move)

                    if (try TYPE_GAME <: Azul catch; false end) && (is_finished(Games[i]) || Games[i].playerToPlay == 0)
                        scores = get_raw_points_round(Games[i])
                        MemoryModule.assign_round(memories[i], scores)
                    end

                    PCR[i] = rand(rngs[Threads.threadid()]) > 1.0/PCR_RATIO
                    ITER[i] = PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"], PCR_REDUCTION) :  LEARNING_PARAMS["MCTS_ITER"]
                    FP[i] = DO_FORCED_PLAYOUTS && !PCR[i]

                    if is_finished(Games[i])
                        winner = get_winner(Games[i])
                        scores = Game.get_delta_points(Games[i])
                        MemoryModule.assign_winner(memories[i], winner, scores)

                    elseif Games[i].playerToPlay == 0
                        # Do random stuff on game 
                        move = get_move_random_stuffs(Games[i], rngs[Threads.threadid()])
                        play!(Games[i], move)
                        keep_subtree!(AZs[i], get_null_move(Games[i]))
                        @assert length(AZs[i].ROOT.children) == 0

                    elseif length(AZs[i].ROOT.children) == 1
                        ITER[i] = PCR[i] ? 2 : div(LEARNING_PARAMS["MCTS_ITER"], PCR_REDUCTION)

                    elseif length(AZs[i].ROOT.children) > 0 && !PCR[i]  # adapt child iterations
                        ITER[i] = min(2*LEARNING_PARAMS["MCTS_ITER"], max(LEARNING_PARAMS["MCTS_ITER"], 2*AZs[i].ROOT.N))    # 400 of if child has been seen a lot, double to add dirichlet effect
                        AZmodule.add_dirichlet!(AZs[i])
                    end

                    Threads.atomic_add!(sum_rounds, 1)  # 1 more round is being done
                end
            end
            time_after += time()
        end

        
        # Put all the memories into the replaybuffer
        size_new = sum([length(M.replayBuffer) for M in memories])
        for M in memories
            for E in M.replayBuffer
                MemoryModule.save!(memory, E)
            end
        end

        GC.gc(true)

        println("EPISODE ", ep, " ", NGAMES, " games done in ", total_time[2], "\tGPU time : ", time_gpu, "\treal_iter : ", real_iter, " /total_iter : ", total_iter, " i.e. ", round(Int,total_time[2]*1000000/real_iter), " us / effective iteration",
                "\n\taverage round : ", 0.1*round(Int, 10*sum_rounds[] / NGAMES),
                "\tlength memory : ", length(memory.replayBuffer), "(new ", size_new, ")")
        println("Time before : ", time_before, " time after: ", time_after)


        if ep%2 == 1
            print("Saving memory...")
            save_to_file(memory)
        end

        # Skip training if EPOCHS is lower than 1 (e.g. : 0.25)
        timeUntilSave = max(timeUntilSave-1, 0)
        if timeUntilSave > 0 && LEARNING_PARAMS["EPOCHS"] < 1 && LEARNING_PARAMS["EPOCHS"] < rand() 
            continue
        end

        ###2 TRAIN
        #=meanSurprise = sum([E.surpriseWeight for E in memory.replayBuffer]) / length(memory.replayBuffer)
        ff(x) = floor(x) + (rand() < x - floor(x))  # ff(3.8) = 3 or 4, 4 with 80% chance 
        f(s) = ff(0.5 + 0.5 * s / meanSurprise)     # in average, retrieves same length as original
        memoryIndexing = vcat([  [E for i in 1:f(E.surpriseWeight)]     for E in memory.replayBuffer]...)
        X = NNet.concatenate_inputs(nn, [E.input for E in memoryIndexing])
        target = hcat([ vcat(E.pi_MCTS, 
                             E.value_final+LEARNING_PARAMS["WEIGHT_vMCTS"]*(E.value_MCTS-E.value_final), 
                             E.points_final+LEARNING_PARAMS["WEIGHT_vMCTS"]*(E.points_MCTS-E.points_final)) 
                        for E in memoryIndexing ]...)
        println("Surprise : mean=", meanSurprise, " size memoryindex = ", length(memoryIndexing))

        data = Flux.DataLoader((X..., target), batchsize = LEARNING_PARAMS["BATCHSIZE"], shuffle=true, partial = false)=#
        data = make_data(memory, nn; surpriseWeight=SURPRISE_WEIGHT)
        GPUArrays.@cached cache begin
            data = data |> gpu
        end

        Flux.testmode!(nn)
        for E in Iterators.reverse(memory.replayBuffer)  if rand(1:10) == 1 && abs(E.value_MCTS-0.5) >= 0.0
            Game.print_input_nn(Games[1], E.input, E.pi_MCTS)
            println("value_MCTS : ", E.value_MCTS, " value_final : ", E.value_final, "\tvalue points_final : ", E.points_final, "\tvalue points_MCTS : ", E.points_MCTS)
            GPUArrays.@cached cache begin
                x = NNet.add_last_dim(nn, E.input) |> gpu
                y = try NNet.predict(nn, x)  |> cpu catch; NNet.predict(nn, x) |> cpu end
                y = NNet.squeeze(nn, y)
            end
            Game.print_input_nn(Games[1], E.input, Flux.softmax(y[1]); print_pos=false)
            println("value NN : ", y[2], "\tsurprise weight : ", E.surpriseWeight, " ", (length(y) >= 3) ? y[3] : " ")
            break
        end end

        function compute_loss()
            L = zeros(4)    #L, Lpi, Lvalue, Lpoints
            for d in data
                GPUArrays.@cached cache begin
                    Ltmp = try loss(nn, d...) catch; println("retry -----"); loss(nn, d...) end
                end
                L[1:length(Ltmp)] += [Ltmp...] / length(data)
            end
            return L
        end

        if timeUntilSave == 0
            loss_value = compute_loss()
            println("loss value : ", loss_value[1], " = (Lpi) ", loss_value[2], " + (Lv) ", loss_value[3], " + (Lpoints) ", loss_value[4])
        else
            loss_value = zeros(4)
        end

        Flux.trainmode!(nn)
        lr_min, lr_max= LEARNING_PARAMS["lr_min"], LEARNING_PARAMS["lr_max"]
        sched = ParameterSchedulers.Triangle(lr_max - lr_min, lr_min, length(data))

        @time for _ in 1:max(1,LEARNING_PARAMS["EPOCHS"])
            for (η, d) in zip(sched, data)
                Flux.adjust!(opt_state, η)
                GPUArrays.@cached cache begin
                    gs = try Flux.gradient( m->loss(m, d...)[1], nn )[1] catch; Flux.gradient( m->loss(m, d...)[1], nn )[1] end
                    Flux.update!(opt_state, Flux.trainable(nn), gs)
                end
            end
        end

        Flux.testmode!(nn)

        Stats.update!(cur_stats, Dict(:ep => 1, :time => time()-time_run, :NNname => nn.name, :games => LEARNING_PARAMS["NGAMES"], :loss => loss_value[1], :loss_pi => loss_value[2], :loss_v => sqrt(loss_value[3]), :loss_points => loss_value[4], :average_round => sum_rounds[] / NGAMES ))

        #SAVING results
        if timeUntilSave == 0
            push!(all_stats, cur_stats)
            Stats.print(all_stats)
            NNet.save(nn, cur_stats.id)
            Stats.save_stats(cur_stats)

            cur_stats = Stats.Stat(cur_stats) # new one

            timeUntilSave = LEARNING_PARAMS["SAVE_EVERY"]
        else
            NNet.save(nn, cur_stats.id)
            Stats.save_stats(cur_stats)

            #nn_cpu = nn |> cpu
            #@save DIRECTORY*"model_"*string(cur_stats.id)*".bson" nn_cpu
            #Stats.save_stats(cur_stats)
        end

        GC.gc(true)
        GPUArrays.unsafe_free!(cache)
        println("EPISODE ", ep, " -- train! done, free memory : ", Base.format_bytes(Sys.free_memory()))

    end

end












#=
function train_parallel_random_game()
    ###use export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE" before running julia -t
    ##1) make N games in parallel and gather data
    ##2) train
    ##3) save / evaluate

    nn = Model.are_all_nn_set() ? Model.initialize_model(:ELO) : Model.initialize_model(:last)

    NGAMES = LEARNING_PARAMS["NGAMES"]
    NGAMES_UNIVERSES = NGAMES*UNIVERSES
    rngs::Vector{AbstractRNG} = [ Random.MersenneTwister(i) for i in 1:Threads.nthreads()]
    memory = MemoryBuffer()
    lk_memory = Threads.ReentrantLock()

    input::Array{Float32, 4} = zeros(Float32, (SHAPE_INPUT..., NGAMES_UNIVERSES))
    output::Array{Float32, 2} = zeros(NUMBER_ACTIONS+1, NGAMES_UNIVERSES)

    @show Flux.GPU_BACKEND
    if Flux.GPU_BACKEND == "AMDGPU"
        @show Base.format_bytes(AMDGPU.soft_memory_limit())
        @show Base.format_bytes(AMDGPU.hard_memory_limit())
        @show Base.format_bytes(AMDGPU.HARD_MEMORY_LIMIT[])
        @show Base.format_bytes(AMDGPU.SOFT_MEMORY_LIMIT[])
    elseif Flux.GPU_BACKEND == "CUDA"
        println("Using CUDA")
    end

    #STATS loading (ELO, games, etc...)
    all_stats = Stats.Stat[]
    for id in 1:10000
        S = Stats.load_stats(id)
        !isnothing(S) || break
        push!(all_stats, S)
    end
    Stats.print(all_stats)
    cur_stats = length(all_stats) > 0 ? Stats.Stat(all_stats[end]) : Stats.Stat()

    # START TRAINING
    for ep in 1:LEARNING_PARAMS["EPISODES"]
        println("\n-----------------------------\nEPISODE ", ep, " -- start")
        Flux.testmode!(nn)
        time_gpu::Float64 = 0
        time_gcgpu::Float64 = 0

        Games::Vector{Game.AbstractRandomGame} = [new_game() for i in 1:NGAMES]
        Games_universes::Array{Game.AbstractRandomGame, 2} = [deepcopy(G) for G in Games, U in 1:UNIVERSES]
        for G in Games_universes
            Random.seed!(G.core_rng, rand(Int))
        end
        Games_after_eval::Array{Game.AbstractRandomGame, 2} = [new_game() for i in 1:NGAMES, U in 1:UNIVERSES]
        AZs::Array{AZAlgorithm, 2} = [AZAlgorithm(Games[i], nn) for i in 1:NGAMES, U in 1:UNIVERSES]
        curs::Array{AZmodule.Node, 2} = [AZmodule.Node(Games[i]) for i in 1:NGAMES, U in 1:UNIVERSES]
        PCR::Vector{Bool} = [ rand(rngs[1], [true, true, true, false]) for i in 1:NGAMES ]
        FP::Vector{Bool} = [ false for i in 1:NGAMES ]

        memories = [MemoryBuffer() for i in 1:NGAMES]

        remaining_games = Threads.Atomic{Int}(NGAMES)
        total_iter, real_iter = 0, 0
        sum_rounds = Threads.Atomic{Int}(0)
        time_run = time()   # en secondes

        cache = GPUArrays.AllocCache()

        total_time = @timed while remaining_games[] > 0
            #ITERATION DE AZ - before eval

            Threads.@threads for (i, u) in [(i, u) for i in 1:NGAMES, u in 1:UNIVERSES]
                if !Game.is_finished(Games[i]) && !AZmodule.is_finished_searching(AZs[i, u].ROOT, PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"],5) : LEARNING_PARAMS["MCTS_ITER"])
                    FP[i] = DO_FORCED_PLAYOUTS && !PCR[i]
                    curs[i, u] = ParallelAZ.before_nn_eval!(AZs[i, u], Games_universes[i, u], input, i + (u-1)*NGAMES, Games_after_eval[i, u] ; forced_playouts=FP[i])
                end
            end

            #EVAL SUR GPU

            time_gpu -= time()

            f((i, u)) = !is_finished(Games[i]) && !AZmodule.is_finished_searching(AZs[i, u].ROOT, PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"],5) : LEARNING_PARAMS["MCTS_ITER"])
            idx_not_finished = [x[1] + (x[2]-1)*NGAMES for x in findall(f,  [(i, u) for i in 1:NGAMES, u in 1:UNIVERSES])]

            if length(idx_not_finished) == 0 #all([Game.is_finished(Games[i]) || AZmodule.is_finished_searching(AZs[i].ROOT, PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"],5) : LEARNING_PARAMS["MCTS_ITER"]) for i in 1:NGAMES])

            else    # Dense
                output[:, idx_not_finished] = nn(input[:, :, :, idx_not_finished])
                total_iter += length(idx_not_finished)
                real_iter += length(idx_not_finished)
            end

            time_gpu += time()

            #ITERATION DE AZ - after eval
            Threads.@threads for i in 1:NGAMES
                if is_finished(Games[i])
                    continue
                end

                for u in 1:UNIVERSES
                    if !AZmodule.is_finished_searching(AZs[i, u].ROOT, PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"],5) : LEARNING_PARAMS["MCTS_ITER"])
                        add_dirichlet = (!PCR[i] && length(AZs[i, u].ROOT.children) == 0)  # Normally happens only on first iteration
                        ParallelAZ.after_nn_eval!(curs[i, u], Games_after_eval[i, u], output, i + (u-1)*NGAMES; rng=rngs[Threads.threadid()], add_dirichlet=add_dirichlet)   #just add dirichlet at root
                    end
                end

                if all([AZmodule.is_finished_searching(AZs[i, u].ROOT,  PCR[i] ? div(LEARNING_PARAMS["MCTS_ITER"],5) : LEARNING_PARAMS["MCTS_ITER"] ) for u in 1:UNIVERSES])
                    temperature::Float32 = PCR[i] ? 0.0f0 : (TEMP_INIT - TEMP_FINAL) * BASE_HALFLIFE^Game.number_played_moves(Games[i]) + TEMP_FINAL
                    pi_MCTS = zeros(Float32, NUMBER_ACTIONS)
                    v_MCTS::Float32 = 0.0
                    for u in 1:UNIVERSES
                        move, pi_MCTS_tmp, v_MCTS_tmp = AZmodule.get_move_temperature(AZs[i, u], Games_universes[i, u], temperature; rng=rngs[Threads.threadid()], forced_playouts=FP[i])
                        pi_MCTS += pi_MCTS_tmp
                        v_MCTS += v_MCTS_tmp
                    end

                    pi_MCTS /= UNIVERSES
                    v_MCTS /= UNIVERSES

                    if temperature == 0.0f0
                        idx = findmax(pi_MCTS)[2]
                    else
                        S = pi_MCTS .^ (1.0f0 / temperature)
                        S /= sum(S)
                        idx = rand(rngs[Threads.threadid()], Categorical(S))
                    end
                    move = Game.get_move_from_idx_move(Games[i], idx)


                    if !PCR[i] #|| AZs[i].ROOT.is_terminal != NOT_TERMINAL
                        MemoryModule.save!(memories[i], Games[i], pi_MCTS, v_MCTS, rngs[Threads.threadid()])
                    end

                    play!(Games[i], move)
                    PCR[i] = rand(rngs[Threads.threadid()], [true, true, true, false])

                    if is_finished(Games[i])
                        Threads.atomic_sub!(remaining_games, 1)
                        winner = get_winner(Games[i])
                        MemoryModule.assign_winner(memories[i], winner, typeof(Games[i]) <: AbstractPointsGame ? Games[i].scores : 0)

                    else
                        for u in 1:UNIVERSES
                            AZs[i, u].ROOT = AZmodule.Node(Games[i])
                            load_from(Games[i], Games_universes[i, u])
                            Random.seed!(Games_universes[i, u].core_rng, rand(Int))
                        end

                    end

                    Threads.atomic_add!(sum_rounds, 1)  # 1 more round is being done
                end
            end
        end

        # Put all the memories into the replaybuffer
        for M in memories
            for E in M.replayBuffer
                if length(memory.replayBuffer) < MemoryModule.SIZE_REPLAYBUFFER
                    push!(memory.replayBuffer, E)
                else
                    MemoryModule.load_from!(E, memory.replayBuffer[memory.elementToWrite])
                end
                memory.elementToWrite = (memory.elementToWrite == MemoryModule.SIZE_REPLAYBUFFER) ? 1 : memory.elementToWrite + 1
            end
        end

        Flux.trainmode!(nn)
        GC.gc(true)

        println("EPISODE ", ep, " ", NGAMES, " games done in ", total_time[2], "\tGPU time : ", time_gpu, "\treal_iter : ", real_iter, " /total_iter : ", total_iter, " i.e. ", 0.001*round(Int,total_time[2]*1000000/real_iter), " ms / iter",
                "\n\taverage round : ", 0.1*round(Int, 10*sum_rounds[] / NGAMES),
                "\tlength memory : ", length(memory.replayBuffer), "(new ", sum([length(M.replayBuffer) for M in memories]), ")")


        ###2 train
        size_new = sum([length(M.replayBuffer) for M in memories])
        last = memory.elementToWrite == 1 ? memory.elementToWrite-1 : length(memory.replayBuffer)
        index_new = last >= size_new ? (last - size_new + 1:last) : vcat(1:last, length(memory.replayBuffer)-size_new+last:length(memory.replayBuffer))

        #X = cat([E.position for E in memory.replayBuffer]..., dims=4)       # TODO PROBLEME ICI : LA RAM DESCEND DE PLUS EN PLUS, 0.2GiB par iteration
        X = [E.position[x, y, k] for x in 1:SHAPE_INPUT[1], y in 1:SHAPE_INPUT[2], k in 1:SHAPE_INPUT[3], E in memory.replayBuffer]
        target = hcat([ vcat(E.pi_MCTS, E.value_final+LEARNING_PARAMS["WEIGHT_vMCTS"]*(E.value_MCTS-E.value_final)) for E in memory.replayBuffer ]...)
#         data_new = Flux.DataLoader((X[:, :, :, index_new], target[:, index_new]), batchsize = LEARNING_PARAMS["BATCHSIZE"], shuffle=true, partial = false)
        data = Flux.DataLoader((X, target), batchsize = LEARNING_PARAMS["BATCHSIZE"], shuffle=true, partial = false)
        GPUArrays.@cached cache if NN_TYPE == "Conv"
            data = data |> gpu
#             data_new = data_new |> gpu
        end

        Flux.testmode!(nn)
        for E in shuffle(memory.replayBuffer)  if abs(E.value_MCTS-0.5) >= 0.0
            Game.print_input_nn(Games[1], E.position, E.pi_MCTS)
            println("value_MCTS : ", E.value_MCTS, " value_final : ", E.value_final)

            #position = Float32[0.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0 0.0; 1.0 1.0 0.0 0.0 0.0; 1.0 1.0 0.0 0.0 0.0;;; 0.0 1.0 0.0 0.0 0.0; 1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
            #position = zeros(Float32, SHAPE_INPUT)
            pi_value, v = Evaluator.predict(nn, E.position; cache=cache)
            Game.print_input_nn(Games[1], E.position, pi_value)
            println("value NN : ", v)
            break
        end end
        Flux.trainmode!(nn)

        function loss(m, x, target)
            y = m(x)
            Lv = Flux.mse(y[NUMBER_ACTIONS+1, :], target[NUMBER_ACTIONS+1, :])
            #Lv = Flux.Losses.binarycrossentropy(y[NUMBER_ACTIONS+1, :], target[NUMBER_ACTIONS+1, :])
            Lpi = Flux.Losses.logitcrossentropy(y[1:NUMBER_ACTIONS, :], target[1:NUMBER_ACTIONS, :]; dims=1)   +    sum(  Flux.Losses.xlogx.(target[1:NUMBER_ACTIONS, :]) ) / size(target)[2]       #Kullback Leibler
            #Lreg = 1f-4*sum(sqnorm, Flux.trainables(m))
            return Lpi+Lv, Lpi, Lv
        end



        loss_value = 0
        loss_pi, loss_v = 0, 0
        for d in data
            L, Lpi, Lv = loss(nn, d...)
            loss_value += L / length(data)
            loss_pi += Lpi / length(data)
            loss_v += Lv / length(data)
        end

        println("loss value : ", loss_value, " = (Lpi) ", loss_pi, " + (Lv) ", loss_v)

        opt_state = Flux.setup( Flux.AdamW( LEARNING_PARAMS["lr_max"] ), nn)
        lr_min, lr_max= LEARNING_PARAMS["lr_min"], LEARNING_PARAMS["lr_max"]
        sched = ParameterSchedulers.Triangle(lr_max - lr_min, lr_min, length(data))

#         GPUArrays.@cached cache begin
            @time for i in 1:LEARNING_PARAMS["EPOCHS"]
                for (η, d) in zip(sched, data)
                    Flux.adjust!(opt_state, η)
                    gs = Flux.gradient( m->loss(m, d...)[1], nn )[1]
                    Flux.update!(opt_state, nn, gs)
                end
            end
#         end

        Flux.testmode!(nn)

        loss_value_after = 0
        loss_pi_after, loss_v_after = 0, 0

        for d in data
            L, Lpi, Lv = loss(nn, d...)
            loss_value_after += L / length(data)
            loss_pi_after += Lpi / length(data)
            loss_v_after += Lv / length(data)
        end

        println("loss value after training : ", loss_value_after, " = (Lpi) ", loss_pi_after, " + (Lv) ", loss_v_after)

        Stats.update!(cur_stats, Dict(:ep => 1, :time => time()-time_run, :games => LEARNING_PARAMS["NGAMES"], :loss => loss_value, :loss_pi => loss_pi, :loss_v => sqrt(loss_v), :average_round => sum_rounds[] / NGAMES ))

        #SAVING results
        if length(all_stats) == 0 || ep%LEARNING_PARAMS["SAVE_EVERY"] == 0
            push!(all_stats, cur_stats)
            Stats.print(all_stats)
            nn_cpu = nn |> cpu
            @save DIRECTORY*"model_"*string(cur_stats.id)*".bson" nn_cpu
            Stats.save_stats(cur_stats)

            cur_stats = Stats.Stat(cur_stats) # new one
        else
            nn_cpu = nn |> cpu
            @save DIRECTORY*"model_"*string(cur_stats.id)*".bson" nn_cpu
            Stats.save_stats(cur_stats)
            
        end

        GC.gc(true)
        AMDGPU.unsafe_free!(cache)
        println("EPISODE ", ep, " -- train! done, free memory : ", Base.format_bytes(Sys.free_memory()))

    end
end
=#


function learn_from_scratch()

    # Some experiments show that if batchsize is too short, the new neural network will have too many holes

    nn = NNet.initialize_model(:new)
    #=nn_transfert = NNet.initialize_model(:ELO)
    nn_new = NNet.initialize_model(:new)
    #nn.trunk = deepcopy(nn_transfert.trunk)
    #nn.v_head = deepcopy(nn_transfert.v_head)
    nn = NNet.BoopNN(deepcopy(nn_transfert.base), deepcopy(nn_transfert.trunk), deepcopy(nn_new.pi_head), deepcopy(nn_transfert.v_head), nn_transfert.name)
    =#
    id = 1
    while isfile(DIRECTORY*"model_"*string(id)*".bson")
        id += 1
    end
    cur_stats = Stats.load_stats(id-1)
    cur_stats.id = id
    cur_stats.NNname = nn.name
    cur_stats.ELO = -1

    Stats.save_stats(cur_stats)

    #Dataset
    memory = MemoryBuffer()
    load_from_file!(memory)
    #=X = [E.position[x, y, k] for x in 1:SHAPE_INPUT[1], y in 1:SHAPE_INPUT[2], k in 1:SHAPE_INPUT[3], E in memory.replayBuffer]
    target = hcat([ vcat(E.pi_MCTS, E.value_final+LEARNING_PARAMS["WEIGHT_vMCTS"]*(E.value_MCTS-E.value_final), E.points_final ) for E in memory.replayBuffer ]...)
    =#
    #=cache = GPUArrays.AllocCache()
    GPUArrays.@cached cache begin
        data = make_data(memory, nn) |> gpu
    end=#
    cache = GPUArrays.AllocCache()
    idx = Random.shuffle(1:length(memory.replayBuffer))
    sep = 7*div(length(memory.replayBuffer), 8)
    trainRange = idx[1:sep]
    testRange =  idx[sep+1:end]

    GPUArrays.@cached cache begin
        dataTrain = make_data(memory, nn; range=trainRange, surpriseWeight=false) |> gpu
        dataTest  = make_data(memory, nn; range=testRange, surpriseWeight=false, shuffle=false) |> gpu
    end

    println("Training on ", length(dataTrain), " and testing on ", length(dataTest), " data")

    #Optimizer
    opt_state = Flux.setup( Flux.AdamW( 1e-3, (0.9, 0.999), 0.025 ), nn)
    #Flux.freeze!(opt_state.base)
    #Flux.freeze!(opt_state.trunk)
    #Flux.freeze!(opt_state.v_head)
    lr_min, lr_max= LEARNING_PARAMS["lr_min"], LEARNING_PARAMS["lr_max"]
    sched = ParameterSchedulers.Triangle(lr_max - lr_min, lr_min, length(dataTrain))

    function compute_loss(data)
        L = zeros(4)    #L, Lpi, Lvalue, Lpoints
        for d in data
            GPUArrays.@cached cache begin
                Ltmp = try loss(nn, d...) catch; println("retry -----"); loss(nn, d...) end
            end
            L[1:length(Ltmp)] += [Ltmp...] / length(data)
        end
        return L
    end

    #Training
    imini, lossMini = 1, 100000
    for i in 1:40
        println("Training...")
        Flux.trainmode!(nn)
        for (η, d) in zip(sched, dataTrain)
        #for d in dataTrain
            #Flux.adjust!(opt_state, η)
            GPUArrays.@cached cache begin
                gs = try Flux.gradient( m->loss(m, d...)[1], nn )[1] catch; Flux.gradient( m->loss(m, d...)[1], nn )[1] end
                Flux.update!(opt_state, Flux.trainable(nn), gs)
            end
        end
        
        Flux.testmode!(nn)
        totalLoss = compute_loss(dataTest)
        if lossMini > totalLoss[1]
            imini = i
            lossMini = totalLoss[1]
        end
        println("i= ", i, "\tloss on testdata: ", totalLoss, " loss on traindata", compute_loss(dataTrain), " mini: ", lossMini)

        cur_stats.loss    = totalLoss[1]
        cur_stats.loss_pi = totalLoss[2]
        cur_stats.loss_v  = sqrt(totalLoss[3])
        cur_stats.loss_points = sqrt.(totalLoss[4])

        for E in [E for E in memory.replayBuffer][testRange] if abs(E.value_MCTS-0.5) >= 0.4
            Game.print_input_nn(BLANK_GAME, E.input, E.pi_MCTS)
            println("value_MCTS : ", E.value_MCTS, " value_final : ", E.value_final, "\tvalue points_final : ", E.points_final, "\tvalue points_MCTS : ", E.points_MCTS)
            GPUArrays.@cached cache begin
                x = NNet.add_last_dim(nn, E.input) |> gpu
                y = try NNet.predict(nn, x)  |> cpu catch; NNet.predict(nn, x) |> cpu end
                y = NNet.squeeze(nn, y)
            end
            Game.print_input_nn(BLANK_GAME, E.input, Flux.softmax(y[1]); print_pos=false)
            println("value NN : ", y[2], "\tsurprise weight : ", E.surpriseWeight)
            break
        end end

        if i == 1 || i == 10 || i == 20 || i == 40
            #push!(all_stats, cur_stats)
            #Stats.print(all_stats)
            NNet.save(nn, cur_stats.id)
            Stats.save_stats(cur_stats)
            cur_stats = Stats.Stat(cur_stats) # new one
        end

        #Stats.save_stats(cur_stats)
        #NNet.save(nn, cur_stats.id)
    end
end


function get_bag(filter)
    bag = zeros(Int, 5)
    for color = 1:5
        bag[color] = round(Int, filter[1, COLORPOS[1, color]])
    end
    return bag
end
function transfert()
    # old azul -> new azul
    memory_old = MemoryBuffer()
    memory = MemoryBuffer()
    load_from_file!(memory_old; fileName="replayBuffer_oldAzul.jls")
    
    for E in memory_old.replayBuffer
        input = E.input
        G = new_game()
        for x in 1:5, y in 1:5, p in 1:NUM_PLAYERS
            G.board[x,y,p] = round(Bool, input[x, y, 5*p-4])
        end
        for p in 1:NUM_PLAYERS
            for line in 1:5
                for y in 1:5
                    if input[line, y, 5*p-3] > 0.0f0
                        qty = input[line, y, 5*p-3] * line
                        color = BOARDCOLOR[line, y]
                        G.pattern[p, line] = (qty, color)
                        break
                    end
                end
            end
        end
        for p in 1:NUM_PLAYERS
            G.scores[p] = round(Int,input[1,1,5*p-2]*10)
            G.floor[p] = round(Int, input[1,1,5*p-1]*7)
            G.hasFirstPlayerTile[p] = round(Bool, input[1,1,5*p])
        end

        G.buffer .= get_bag(input[:, :, 5*NUM_PLAYERS+1]*5.0)
        for line in 1:CENTER
            G.factory[line, :] .= get_bag(input[:, :, 5*NUM_PLAYERS+1+line] * 4.0)
        end
        G.bag = get_bag(input[:, :, end]*20.0)

        input_new = get_input_for_nn(G)


        elem = MemoryModule.Element(input_new, E.playerToPlay, E.value_MCTS, E.value_final, E.points_MCTS, E.points_final, E.pi_MCTS, E.surpriseWeight)
        MemoryModule.save!(memory, elem)

    end

    println("size memory : ", length(memory.replayBuffer))
    save_to_file(memory)

end

end



