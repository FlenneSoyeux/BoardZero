"""
    multi.jl

    extension of Arena.jl
    functions for simultaneous PvP matches :
        * MCTS vs MCTS with different iterations in order to get ELOS of all iterations
        * AZ(400 iterations) vs MCTS(all different set iterations with set ELOs) to get ELO of AZ
        * AZ(400 iterations, previous versions with set ELO) vs AZ(400 iterations, new versions) to get ELO of new AZ
"""

#=
function MCTS_vs_MCTS_arena(iterations::Vector{Int}; NGAMES::Int = 100)
    # Evaluate MCTS1 vs MCTS2 in a threaded way
    # To be used to evaluate the UCB constant for instance
    scores = [0, 0, 0] #wins1, wins2, draws

    for iter in iterations
        rngs::Vector{AbstractRNG} = [ Random.MersenneTwister(iter + i) for i in 1:Threads.nthreads()]
        wins1 = Threads.Atomic{Int}(0) #score for 1 ; score for 2 ; draws
        wins2 = Threads.Atomic{Int}(0)
        draws = Threads.Atomic{Int}(0)

        Threads.@threads for game in 1:NGAMES
            G = new_game()
            G.playerToPlay = 1 + (game%2)
            mcts_a = MCTSAlgorithm(G, iter, 0.0)
            mcts_b = MCTSAlgorithm(G, iter, 0.0)
            while !is_finished(G)
                move = (G.playerToPlay == 1) ? get_move(mcts_a, G, rngs[Threads.threadid()]) : get_move(mcts_b, G, rngs[Threads.threadid()])
                play!(G, move)
                keep_subtree!(mcts_a, move)
                keep_subtree!(mcts_b, move)
            end

            winner = get_winner(G)
            if winner == 0
                Threads.atomic_add!(draws, 1)
            elseif winner == 1
                Threads.atomic_add!(wins1, 1)
            else
                Threads.atomic_add!(wins2, 1)
            end
        end

        println("FINI AVEC iter=", iter, "\tscore : ", wins1[], " vs ", wins2[], " (draws :", draws[], ")")
        scores[1] += wins1[]
        scores[2] += wins2[]
        scores[3] += draws[]
    end

    println("Total : ", scores, "ELO diff : ", get_elo(scores[1] / (scores[1]+scores[2])))


end=#

"""
    MCTS_arena()
    Compute all ELOs of all iterations given in params.jl in keys of PARAMS_MCTS_ELO
    create a dictionary (x, y) => win_rate(x vs y)
    and gives relative ELOs
    if there is one with <= 2 iterations : this one will have 0 elo
"""
function MCTS_arena()
    # What is the ELO of a MCTS of a given iteration ?
    iterations = [k for k in keys(PARAMS_MCTS_ELO)] #[50, 400, 800, 1600, 6400, 12800]
    NGAMES = 50   #each i plays vs j NGAMES times
    scores = Dict([(x,y) => 0.0f0 for x in iterations, y in iterations])
    lk_scores = Threads.ReentrantLock()
    rngs::Vector{AbstractRNG} = [ Random.Xoshiro(rand(Int)) for i in 1:Threads.nthreads()]
    progress_bar = Progress( (length(iterations)^2 - length(iterations))*NGAMES )
    Threads.@threads for (iter0, iter1, i) in shuffle([(iter0, iter1, i) for iter0 in iterations, iter1 in iterations, i in 1:NGAMES])
        if iter0 != iter1
            s = make_game(iter0, iter1, rngs[Threads.threadid()])
            lock(lk_scores) do
                scores[(iter0, iter1)] += s / NGAMES /2
                scores[(iter1, iter0)] += ( 1.0-s ) / NGAMES /2
                next!(progress_bar)

            end
        end
    end
    finish!(progress_bar)

    ELOs = ELO.relative_elos(scores)
    mini = minimum(keys(ELOs))
    val = ELOs[mini]
    if mini <= 2
        for x in keys(ELOs)
            ELOs[x] = ELOs[x] - val
        end
    end

    println(ELOs)
    display(pretty_table( [scores[(x,y)] for x in iterations, y in iterations] ; header=iterations, row_labels=iterations ))
    display(pretty_table( ELOs ))
end

function make_game(iter0, iter1, rng) :: Float32
    # Returns 0, 0.5 or 1 (score of player0)
    G = Game.new_game()
    mcts0 = MCTSAlgorithm(G, iter0, 0.0)
    mcts1 = MCTSAlgorithm(G, iter1, 0.0)
    while !is_finished(G)
        move = (G.playerToPlay == 1) ? get_move(mcts0, G, rng) : get_move(mcts1, G, rng)
        Game.play!(G, move)
        keep_subtree!(mcts0, move)
        keep_subtree!(mcts1, move)
    end

    w = Game.get_winner(G)
    if w == 0   # DRAW
        return 0.5
    elseif w == 1   # WIN for player0
        return 1
    else            # LOSE for player0
        return 0
    end
end

"""
    AZ_vs_MCTS_arena()
    Get ELO of all AZ neural networks with matches with the MCTS algorithm
"""

function set_first_to_zero()
    file = BSON.load(DIRECTORY*"stats_1.bson")
    file[:ELO] = 0
    BSON.bson(DIRECTORY*"stats_1.bson", file)
end

function AZ_vs_MCTS_arena()
    # Open a NN with no ELO (or =-1) and set it
    id = 1
    has_computed = false
    while isfile(DIRECTORY*"model_"*string(id)*".bson")
        dict = BSON.load(DIRECTORY*"stats_"*string(id)*".bson")
        if dict[:ELO] == -1 #Il faut mettre un elo !
            elo = AZ_vs_MCTS_arena_util(id)
            dict[:ELO] = elo
            BSON.bson(DIRECTORY*"stats_"*string(id)*".bson", dict)
            has_computed = true
        end
        id += 1
    end
    Stats.print()
end



function AZ_vs_MCTS_arena_util(id)
    BSON.@load DIRECTORY*"model_"*string(id)*".bson" nn_cpu
    nn = nn_cpu |> DEVICE
    println("Calcul de l'ELO du modèle ", id)

    results = Dict()
    N = 0
    sum_elo = 0
    for x in keys(PARAMS_MCTS_ELO), y in keys(PARAMS_MCTS_ELO)
        results[(x, y)] = ELO.winrate(PARAMS_MCTS_ELO[x], PARAMS_MCTS_ELO[y])
    end
    for iter in sort([iter for iter in keys(PARAMS_MCTS_ELO)])
        w = AZ_vs_MCTS_multi(nn, PARAMS_MCTS_ELO[iter])
        results[(-1, iter)] = w
        results[(iter, -1)] = 1.0-w
        N += 1
        sum_elo += PARAMS_MCTS_ELO[iter]
        if w < 0.1
            break
        end
    end
    ELOs = ELO.relative_elos(results)
    mean_ELOs_before = sum_elo/N
    mean_ELOs_after = sum([get(ELOs, iter, 0) for iter in keys(PARAMS_MCTS_ELO)]) / N
    elo = max(0, ELOs[-1] - mean_ELOs_after + mean_ELOs_before)

    println("Modèle ", id, " elo: ", elo)
    return elo
end

function AZ_vs_MCTS_multi(nn::Chain, mcts_iteration::Int)
    if typeof(new_game()) <: AbstractRandomGame
        return AZ_vs_MCTS_multi_random_game(nn, mcts_iteration)
    end
    NGAMES = 50
    games = [new_game() for i in 1:NGAMES]
    mcts_player = [1+(i%2 == 0) for i in 1:NGAMES]   #true : mcts plays ; false : AZ plays
    MCTSs = [MCTSAlgorithm(games[i], mcts_iteration, 0.0) for i in 1:NGAMES]
    AZs = [AZAlgorithm(games[i], nn, 400, 0.0) for i in 1:NGAMES]
    rngs::Vector{AbstractRNG} = [ Random.Xoshiro(rand(Int)) for i in 1:Threads.nthreads()]

    Games_after_eval::Vector{Game.AbstractGame} = [new_game() for i in 1:NGAMES]
    curs::Vector{AZmodule.Node} = [AZmodule.Node(games[i]) for i in 1:NGAMES]
    input::Array{Float32, 4} = zeros(SHAPE_INPUT..., NGAMES)
    output::Array{Float32, 2} = zeros(NUMBER_ACTIONS+1, NGAMES)

    wins, draws, losses = 0, 0, 0

    progress_bar = Progress( max(100, NUM_CELLS) )
    cache = GPUArrays.AllocCache()

    #MCTS makes a turn on half games
    Threads.@threads for i in 1:NGAMES
        if mcts_player[i] == 1
            while games[i].playerToPlay == 1
                move = get_move(MCTSs[i], games[i], rngs[Threads.threadid()])
                play!(games[i], move)
                keep_subtree!(MCTSs[i], move)
                keep_subtree!(AZs[i], move)
            end
        end
    end


    GC.gc(false)
    last_text = ""

    remaining_games = NGAMES
    while remaining_games > 0
        #PLAY all ALPHAZERO moves
        while any([!is_finished(games[i]) && games[i].playerToPlay == 3-mcts_player[i] for i in 1:NGAMES])
            # a) BEFORE EVAL
            Threads.@threads for i in 1:NGAMES
                if is_finished(games[i]) || games[i].playerToPlay == mcts_player[i]
                    continue
                end
                curs[i] = ParallelAZ.before_nn_eval!(AZs[i], games[i], Games_after_eval[i])
                NNet.inplace_change!(nn[p], input[p], i, Game.get_input_for_nn(Games_after_eval[i]))
            end

            # b) EVAL
            output = Inference.predict(nn, input)

            # c) AFTER EVAL
            Threads.@threads for i in 1:NGAMES
                if is_finished(games[i]) || games[i].playerToPlay == mcts_player[i]
                    continue

                elseif !AZmodule.is_finished_searching_exploitation_mode(AZs[i].ROOT, 400)
                    ParallelAZ.after_nn_eval!(curs[i], Games_after_eval[i], output, i; rng=rngs[Threads.threadid()])
                end

                if AZmodule.is_finished_searching_exploitation_mode(AZs[i].ROOT, 400)
#                     move  = AZmodule.get_best_move(AZs[i], games[i], rngs[Threads.threadid()])
                    move, = AZmodule.get_move_temperature(AZs[i], games[i], 0.0f0; rng=rngs[Threads.threadid()])
                    play!(games[i], move)
                    keep_subtree!(AZs[i], move)
                    keep_subtree!(MCTSs[i], move)

                end
            end
        end


        wins, draws, losses = get_wins_draws_losses(games, mcts_player, NGAMES)
        progress_bar.core.desc =   "W: "*string(wins)*" D: "*string(draws)*" L: "*string(losses)
        ProgressMeter.next!(progress_bar)
        sleep(0.01)
        GC.gc(true)

        #PLAY all MCTS moves
        while any([  !is_finished(games[i]) && games[i].playerToPlay == mcts_player[i]  for i in 1:NGAMES])     #while some alphazero moves must be done
            Threads.@threads for i in 1:NGAMES
                if Game.is_finished(games[i]) || games[i].playerToPlay == 3-mcts_player[i]
                    continue
                end
                move = get_move(MCTSs[i], games[i], rngs[Threads.threadid()])
                play!(games[i], move)
                keep_subtree!(AZs[i], move)
                keep_subtree!(MCTSs[i], move)
            end
        end

        wins, draws, losses = get_wins_draws_losses(games, mcts_player, NGAMES)
        progress_bar.core.desc =  "W: "*string(wins)*" D: "*string(draws)*" L: "*string(losses)
        ProgressMeter.next!(progress_bar)
        sleep(0.01)
        remaining_games = sum([!is_finished(G) for G in games])
    end

    ProgressMeter.finish!(progress_bar)

    wins, draws, losses = get_wins_draws_losses(games, mcts_player, NGAMES)
    return (wins + draws/2) / NGAMES
end

#=
function AZ_vs_MCTS_multi_random_game(nn, mcts_iteration)
    NGAMES = 50
    Games = [new_game() for i in 1:NGAMES]
    Games_universes::Array{Game.AbstractRandomGame, 2} = [deepcopy(G) for G in Games, U in 1:UNIVERSES]
    mcts_player = [1+(i%2 == 0) for i in 1:NGAMES]   #true : mcts plays ; false : AZ plays
    MCTSs = [MCTSAlgorithm(Games[i], mcts_iteration, 0.0) for i in 1:NGAMES]
    AZs = [AZAlgorithm(Games[i], nn, 400, 0.0, true) for i in 1:NGAMES, u in 1:UNIVERSES]
    rngs::Vector{AbstractRNG} = [ Random.MersenneTwister(i) for i in 1:Threads.nthreads()]

    Games_after_eval::Array{Game.AbstractGame, 2} = [new_game() for i in 1:NGAMES, U in 1:UNIVERSES]
    curs::Array{AZmodule.Node, 2} = [AZmodule.Node(Games[i]) for i in 1:NGAMES, U in 1:UNIVERSES]
    input::Array{Float32, 4} = zeros(SHAPE_INPUT..., NGAMES*UNIVERSES)
    output::Array{Float32, 2} = zeros(NUMBER_ACTIONS+1, NGAMES*UNIVERSES)

    wins, draws, losses = 0, 0, 0

    progress_bar = Progress( max(100, NUM_CELLS) )
    cache = GPUArrays.AllocCache()

    #MCTS makes a turn on half games
    Threads.@threads for i in 1:NGAMES
        if mcts_player[i] == 1
            while Games[i].playerToPlay == 1
                move = get_move(MCTSs[i], Games[i], rngs[Threads.threadid()])
                play!(Games[i], move)
            end
        end
    end

    GC.gc(false)
    last_text = ""

    remaining_games = NGAMES
    while remaining_games > 0
        #PLAY all ALPHAZERO moves
        for i in 1:NGAMES, u in 1:UNIVERSES
            if !is_finished(Games[i])
                AZs[i, u].ROOT = AZmodule.Node(Games[i])
                load_from(Games[i], Games_universes[i, u])
                Random.seed!(Games_universes[i, u].core_rng, rand(Int))
            end
        end

        while any([!is_finished(Games[i]) && Games[i].playerToPlay == 3-mcts_player[i] for i in 1:NGAMES])
            # a) BEFORE EVAL
            Threads.@threads for (i, u) in [(i, u) for i in 1:NGAMES, u in 1:UNIVERSES]
                if !is_finished(Games[i]) && Games[i].playerToPlay == 3-mcts_player[i] && !AZmodule.is_finished_searching(AZs[i, u].ROOT, 400)
                    curs[i, u] = ParallelAZ.before_nn_eval!(AZs[i, u], Games_universes[i, u], input, i + (u-1)*NGAMES, Games_after_eval[i, u])
                end
            end

            f((i, u)) = !is_finished(Games[i]) && Games[i].playerToPlay == 3-mcts_player[i] && !AZmodule.is_finished_searching(AZs[i, u].ROOT, 400)
            idx_not_finished = [x[1] + (x[2]-1)*NGAMES for x in findall(f,  [(i, u) for i in 1:NGAMES, u in 1:UNIVERSES])]

            # b) EVAL
            if length(idx_not_finished) == 0

            else
                output[:, idx_not_finished] = nn(input[:, :, :, idx_not_finished])
            end


            # c) AFTER EVAL
            Threads.@threads for i in 1:NGAMES
                if is_finished(Games[i]) || Games[i].playerToPlay == mcts_player[i]
                    continue
                end

                for u in 1:UNIVERSES
                    if !AZmodule.is_finished_searching(AZs[i, u].ROOT, 400)
                        ParallelAZ.after_nn_eval!(curs[i, u], Games_after_eval[i,u], output, i + (u-1)*NGAMES; rng=rngs[Threads.threadid()])
                    end
                end

                if all([AZmodule.is_finished_searching(AZs[i, u].ROOT, 400) for u in 1:UNIVERSES])
                    pi_MCTS = zeros(Float32, NUMBER_ACTIONS)
                    for u in 1:UNIVERSES
                        move, pi_MCTS_tmp, v_MCTS_tmp = ParallelAZ.get_move_temperature(AZs[i, u], Games_universes[i, u], 0.0f0; rng=rngs[Threads.threadid()])
                        pi_MCTS += pi_MCTS_tmp
                    end
                    idx = findmax(pi_MCTS)[2]
                    move = Game.get_move_from_idx_move(Games[i], idx)
                    play!(Games[i], move)

                    for u in 1:UNIVERSES
                        AZs[i, u].ROOT = AZmodule.Node(Games[i])
                        load_from(Games[i], Games_universes[i, u])
                        Random.seed!(Games_universes[i, u].core_rng, rand(Int))
                    end
                end
            end
        end


        wins, draws, losses = get_wins_draws_losses(Games, mcts_player, NGAMES)
        progress_bar.core.desc =   "W: "*string(wins)*" D: "*string(draws)*" L: "*string(losses)
        ProgressMeter.next!(progress_bar)
        sleep(0.01)
        GC.gc(false)


        #PLAY all MCTS moves
        while any([  !is_finished(Games[i]) && Games[i].playerToPlay == mcts_player[i]  for i in 1:NGAMES])     #while some alphazero moves must be done
            Threads.@threads for i in 1:NGAMES
                if Game.is_finished(Games[i]) || Games[i].playerToPlay == 3-mcts_player[i]
                    continue
                end
                move = get_move(MCTSs[i], Games[i], rngs[Threads.threadid()])
                play!(Games[i], move)
            end
        end

        wins, draws, losses = get_wins_draws_losses(Games, mcts_player, NGAMES)
        progress_bar.core.desc =  "W: "*string(wins)*" D: "*string(draws)*" L: "*string(losses)
        ProgressMeter.next!(progress_bar)
        sleep(0.01)
        remaining_games = sum([!is_finished(G) for G in Games])
    end

    ProgressMeter.finish!(progress_bar)

    wins, draws, losses = get_wins_draws_losses(Games, mcts_player, NGAMES)
    return wins + draws/2
end=#


function get_wins_draws_losses(games, mcts_player::Vector{Int}, NGAMES)
    wins, draws, losses = 0, 0, 0
    for i in 1:NGAMES
        if !Game.is_finished(games[i])
            continue
        end
        w = Game.get_winner(games[i])
        wins += (w != mcts_player[i])
        draws += (w==0)
        losses += (w == mcts_player[i])
    end
    return wins, draws, losses
end

"""
    AZ_arena(list_ids = [], NGAMES=100)
    Make matches between all NN from list_ids
    Set all ELOS : if already set, it will change a little bit
"""


function get_wins_draws_losses(games)   #for player 1
    wins, draws, losses = 0,0,0
    for G in games
        winner = get_winner(G)
        draws += (winner == 0)
        wins += (winner == 1)
        losses += (winner == 2)
    end
    return wins, draws, losses
end



function AZ_vs_AZ_match(nna::AbstractNN, nnb::AbstractNN; NGAMES=100)
    #makes NGAMES games between ida and idb
    ITER = 400

    nn = [nna, nnb]
    games = [new_game() for i in 1:NGAMES]
    for i in 1:NGAMES
        games[i].playerToPlay = 1 + (i%2)
    end

    progress_bar = Progress( NGAMES )

    AZs = [AZAlgorithm(games[i], nn[p], ITER, 0.0) for i in 1:NGAMES, p in 1:2]
    rngs::Vector{AbstractRNG} = [ Random.Xoshiro(i) for i in 1:Threads.nthreads()]

    Games_after_eval::Vector{Game.AbstractGame} = [new_game() for i in 1:NGAMES]
    curs::Vector{AZmodule.Node} = [AZmodule.Node(games[i]) for i in 1:NGAMES]
    
    input = [NNet.get_input_shape(nna, NGAMES), NNet.get_input_shape(nnb, NGAMES)]
    output = Any[0, 0]

    wins, draws, losses = 0, 0, 0

    cache = GPUArrays.AllocCache()

    temperature = 1.0f0
    iter = 0
    while any([!is_finished(G) for G in games])
        for p in 1:2    # player 1 (nna) and player 2 (nnb) play
            iter += 1
            while any([!is_finished(games[i]) && games[i].playerToPlay == p for i in 1:NGAMES])
                #before nn
                Threads.@threads for i in 1:NGAMES
                    if is_finished(games[i]) || games[i].playerToPlay != p || AZmodule.is_finished_searching_exploitation_mode(AZs[i, p].ROOT, ITER)
                        continue
                    end
                    curs[i] = ParallelAZ.before_nn_eval!(AZs[i, p], games[i], Games_after_eval[i])
                    if !is_finished(Games_after_eval[i])
                        NNet.inplace_change!(nn[p], input[p], i, Game.get_input_for_nn(Games_after_eval[i]))
                    end
                end

                
                if all([is_finished(games[i]) || games[i].playerToPlay != p || AZmodule.is_finished_searching_exploitation_mode(AZs[i, p].ROOT, ITER) for i in 1:NGAMES])

                else
                    GPUArrays.@cached cache begin
                        output[p] = try 
                                NNet.predict(nn[p], input[p] |> gpu) |> cpu
                            catch
                                println("--------------------------------* FAILED - retry *-------------------------------- ")
                                NNet.predict(nn[p], input[p] |> gpu) |> cpu
                            end
                    end
                end
                #NN

                #after nn
                Threads.@threads for i in 1:NGAMES
                    if is_finished(games[i]) || games[i].playerToPlay != p
                        continue

                    elseif !AZmodule.is_finished_searching_exploitation_mode(AZs[i, p].ROOT, ITER )
                        ParallelAZ.after_nn_eval!(curs[i], Games_after_eval[i], games[i], output[p], i; rng=rngs[Threads.threadid()])
                    end

                    if AZmodule.is_finished_searching_exploitation_mode(AZs[i, p].ROOT, ITER )
                        move,  = AZmodule.get_move_temperature(AZs[i, p], games[i], temperature; rng=rngs[Threads.threadid()])
                        play!(games[i], move)
                        keep_subtree!(AZs[i, 1], move)
                        keep_subtree!(AZs[i, 2], move)

                        while !is_finished(games[i]) && games[i].playerToPlay == 0
                            move = get_move_random_stuffs(games[i], rngs[Threads.threadid()])
                            play!(games[i], move)
                            keep_subtree!(AZs[i, 1], move)
                            keep_subtree!(AZs[i, 2], move)
                        end

                        #if is_finished(games[i])
                        #    next!(progress_bar, desc="Moves: "*string(iter))
                        #end
                    end
                end
            end

            temperature = max(temperature - 0.05f0, 0.2f0)
            ProgressMeter.update!(progress_bar, sum([is_finished(G) for G in games]); desc="Moves: "*string(iter))
        end
    end
    finish!(progress_bar)

    wins, draws, losses = get_wins_draws_losses(games)
    return (wins + draws/2) / NGAMES
end


mean(arr) = sum(arr) / length(arr)

function AZ_arena(list_ids = []; NGAMES=100)
    nn = Dict([id => NNet.initialize_model(id) for id in list_ids])
    ELOs = Dict([ id => BSON.load(DIRECTORY*"stats_"*string(id)*".bson")[:ELO] for id in list_ids])
    for id in list_ids 
        if id == 1
            ELOs[id] = 0
        end
    end
    search_elo = Dict([id => (ELOs[id] == -1) for id in list_ids])
    mean_ELOs_before = mean([ ELOs[x] for x in list_ids if !search_elo[x] ])

    results = Dict{ Tuple{Int, Int}, Float64}()
    if isfile("_tmp_results.bson")
        println("WARNING : _tmp_results is present")
        BSON.@load "_tmp_results.bson" results
    end
    

    for i in list_ids
        results[(i, i)] = 0.5
    end
    for ia in list_ids, ib in list_ids
        if ia < ib && !((ia, ib) in keys(results))
            if search_elo[ia] | search_elo[ib]
                # make matches between ida & idb, one of them is not already set
                println("Match between ida = ", ia, " and ", ib)
                results[(ia, ib)] = AZ_vs_AZ_match(nn[ia], nn[ib]; NGAMES=NGAMES)
                results[(ib, ia)] = 1-results[(ia, ib)]
            else
                results[(ia, ib)] = ELO.winrate(ELOs[ia], ELOs[ib])
                results[(ib, ia)] = ELO.winrate(ELOs[ib], ELOs[ia])
            end

            BSON.@save "_tmp_results.bson" results
            ELOs_tmp = ELO.relative_elos(results)
            mean_ELOs_after = mean([ ELOs_tmp[x] for x in list_ids if !search_elo[x] ])
            for x in keys(ELOs)
                ELOs_tmp[x] = ELOs_tmp[x] - mean_ELOs_after + mean_ELOs_before
            end
            ELO.printing(results, ELOs_tmp)
        end
    end

    println("FIN")

    #find ELOs
    ELOs = ELO.relative_elos(results)
    mean_ELOs_after = mean([ ELOs[x] for x in list_ids if !search_elo[x] ])
    for x in keys(ELOs)
        ELOs[x] = ELOs[x] - mean_ELOs_after + mean_ELOs_before
    end

    ELO.printing(results, ELOs)

    for id in list_ids
        dict = BSON.load(DIRECTORY*"stats_"*string(id)*".bson")
        dict[:ELO] = round(Int, ELOs[id])
        BSON.bson(DIRECTORY*"stats_"*string(id)*".bson", dict)
    end

    rm("_tmp_results.bson")

    Stats.print()
end


#=
function AZ_vs_AZ_match_random(nna::Chain, nnb::Chain, NGAMES=100)
    #makes NGAMES games between ida and idb

    nn = [nna, nnb]
    Games = [new_game() for i in 1:NGAMES]
    Games_universes = [new_game() for i in 1:NGAMES, u in 1:UNIVERSES]
    for i in 1:NGAMES
        Games[i].playerToPlay = 1 + (i%2)
    end

    progress_bar = Progress( NGAMES )

    AZs = [AZAlgorithm(Games[i], nn[p], 400, 0.0, true) for i in 1:NGAMES, p in 1:2, u in 1:UNIVERSES]
    rngs::Vector{AbstractRNG} = [ Random.MersenneTwister(i) for i in 1:Threads.nthreads()]

    Games_after_eval::Array{Game.AbstractRandomGame, 2} = [new_game() for i in 1:NGAMES, u in 1:UNIVERSES]
    curs::Array{AZmodule.Node, 2} = [AZmodule.Node(Games[i]) for i in 1:NGAMES, u in 1:UNIVERSES]
    input = [zeros(Float32, SHAPE_INPUT..., NGAMES*UNIVERSES), zeros(Float32, SHAPE_INPUT..., NGAMES*UNIVERSES)]
    output = [zeros(Float32, NUMBER_ACTIONS+1, NGAMES*UNIVERSES), zeros(Float32, NUMBER_ACTIONS+1, NGAMES*UNIVERSES)]

    wins, draws, losses = 0, 0, 0

    cache = GPUArrays.AllocCache()

    temperature = 1.0f0
    iter = 0
    while any([!is_finished(G) for G in Games])
        for p in 1:2    # player 1 (nna) and player 2 (nnb) play

            for i in 1:NGAMES, u in 1:UNIVERSES
                AZs[i, p, u].ROOT = AZmodule.Node(Games[i])
                load_from(Games[i], Games_universes[i, u])
                Random.seed!(Games_universes[i, u].core_rng, rand(Int))
            end

            iter += 1
            while any([!is_finished(Games[i]) && Games[i].playerToPlay == p for i in 1:NGAMES])
                #before nn
                Threads.@threads for (i, u) in [(i, u) for i in 1:NGAMES, u in 1:UNIVERSES]
                    if !is_finished(Games[i]) && Games[i].playerToPlay == p && !AZmodule.is_finished_searching(AZs[i, p, u].ROOT, 400)
                        curs[i, u] = ParallelAZ.before_nn_eval!(AZs[i, p, u], Games_universes[i, u], input[p], i + (u-1)*NGAMES, Games_after_eval[i, u])
                    end

                end

                f((i, u)) = !is_finished(Games[i]) && Games[i].playerToPlay == p && !AZmodule.is_finished_searching(AZs[i, p, u].ROOT, 400)
                idx_not_finished = [x[1] + (x[2]-1)*NGAMES for x in findall(f,  [(i, u) for i in 1:NGAMES, u in 1:UNIVERSES])]

                # b) EVAL
                if length(idx_not_finished) == 0

                else
                    output[p][:, idx_not_finished] = nn[p](input[p][:, :, :, idx_not_finished])
                end
                #NN

                #after nn
                Threads.@threads for i in 1:NGAMES
                    if is_finished(Games[i]) || Games[i].playerToPlay != p
                        continue
                    end

                    for u in 1:UNIVERSES
                        if !AZmodule.is_finished_searching(AZs[i, p, u].ROOT, 400)
                            ParallelAZ.after_nn_eval!(curs[i, u], Games_after_eval[i,u], output[p], i + (u-1)*NGAMES; rng=rngs[Threads.threadid()])
                        end
                    end

                    if all([AZmodule.is_finished_searching(AZs[i, p, u].ROOT, 400) for u in 1:UNIVERSES])
                        pi_MCTS = zeros(Float32, NUMBER_ACTIONS)
                        for u in 1:UNIVERSES
                            move, pi_MCTS_tmp, v_MCTS_tmp = ParallelAZ.get_move_temperature(AZs[i, p, u], Games_universes[i, u], 0.0f0; rng=rngs[Threads.threadid()])
                            pi_MCTS += pi_MCTS_tmp
                        end

                        if temperature == 0.0f0
                            idx = findmax(pi_MCTS)[2]
                        else
                            pi_MCTS = pi_MCTS .^ (1.0f0 / temperature)
                            pi_MCTS /= sum(pi_MCTS)
                            idx = rand(rngs[Threads.threadid()], Categorical(pi_MCTS))
                        end
                        move = Game.get_move_from_idx_move(Games[i], idx)
                        play!(Games[i], move)

                        for u in 1:UNIVERSES
                            AZs[i, p, u].ROOT = AZmodule.Node(Games[i])
                            load_from(Games[i], Games_universes[i, u])
                            Random.seed!(Games_universes[i, u].core_rng, rand(Int))
                        end

                        if is_finished(Games[i])
                            next!(progress_bar, desc="Moves: "*string(iter))
                        end
                    end
                end
            end

            temperature = max(temperature - 0.05f0, 0.2f0)
        end
    end
    finish!(progress_bar)

    wins, draws, losses = get_wins_draws_losses(Games)
    return (wins + draws/2) / NGAMES

end
=#