module AZmodule

export AZAlgorithm

using ..Game, ..Algorithm, ..NNet

using Flux, GPUArrays
using Random
using StatsBase
using Distributions: Categorical, Dirichlet


include("../params.jl")

const CPUCT::Float32 = 1.25

mutable struct Node
    justPlayedMove::AbstractMove    # move to be played
    justPlayedPlayer::Int # 1 or 2
    w::Float32
    points::Vector{Float32}
    N::Int
    pi_value::Float32   # pi value given by NN
    pi_dirichlet::Float32
    v_initial::Vector{Float32}  # first value (directly given by NN) or -1 : for playerToPlay and not justPlayedPlayer!!!
    children::Vector{Node}
    parent::Union{Nothing,Node}
    
    Node(move::AbstractMove, playerOfMove::Int, pi_value::Float32, parent) = new(move, playerOfMove, 0, [0 for _ in 1:LENGTH_POINTS], 0, pi_value, 0, [-1], Vector{Node}(), parent )
    Node(G::AbstractGame) =  new(get_null_move(G), 1, 0,  [0 for _ in 1:LENGTH_POINTS], 0, -1, 0, [-1], Vector{Node}(), nothing)  #Whatever if justPlayedPlayer is equal to playerToPlay, as only future counts, we can do as if players just shifted
    Node(move::AbstractMove, playerOfMove::Int) =  new(move, playerOfMove, 0,  [0 for _ in 1:LENGTH_POINTS], 0, -1, 0, [-1], Vector{Node}(), nothing)
end

function PUCT(n::Node, use_fpu::Bool=true, use_dirichlet::Bool=true) :: Float32
    q::Float32 = 0.0f0
    @static if TYPE_GAME <: AbstractPointsGame
        if n.N == 0 && n.parent.justPlayedPlayer == 0
            q = (0.5 * n.parent.w + 0.5*points_to_utility(BLANK_GAME, n.parent.points)) / n.parent.N
            q = (n.justPlayedPlayer == 1) ? q - use_fpu * FPU : 1.0 - q - use_fpu * FPU
        elseif n.N == 0
            q = (0.5 * n.parent.w + 0.5*points_to_utility(BLANK_GAME, n.parent.points)) / n.parent.N
            q = (n.parent.justPlayedPlayer == n.justPlayedPlayer) ? q - use_fpu * FPU : 1.0 - q - use_fpu * FPU
        else
            q = ( 0.5*n.w + 0.5*points_to_utility(BLANK_GAME, n.points) ) / n.N
        end
    else
        if n.N == 0 && n.parent.justPlayedPlayer == 0
            q = (n.justPlayedPlayer == 1) ? n.parent.w / n.parent.N - use_fpu * FPU : 1.0 - n.parent.w / n.parent.N - use_fpu * FPU # https://www.reddit.com/r/cbaduk/comments/8j5x3w/first_play_urgency_fpu_parameter_in_alpha_zero/
        elseif n.N == 0
            q = (n.parent.justPlayedPlayer == n.justPlayedPlayer) ? n.parent.w / n.parent.N - use_fpu * FPU : 1.0 - n.parent.w / n.parent.N - use_fpu * FPU
        else
            q = n.w/n.N
        end
    end
    u = (use_dirichlet ? n.pi_value + LEARNING_PARAMS["WEIGHT_DIRICHLET"] * (n.pi_dirichlet - n.pi_value) : n.pi_value) * sqrt(n.parent.N) / (1+n.N)

    return q + CPUCT * u
end

function select_child(n::Node) :: Node
    if n.children[1].justPlayedPlayer == 0
        i = mod(n.N, length(n.children))+1
        return n.children[i]
    end

    best_value = -INF
    best_child::Node = n.children[1]
    for child in n.children
        cur = PUCT(child, true, false)
        if(cur > best_value)
            best_value = cur
            best_child = child
        end
    end
    return best_child
end

function select_child_forced_playouts(n::Node) :: Node
    @assert isnothing(n.parent) # n should be ROOT
    @assert n.children[1].justPlayedPlayer != 0

    best_value = -INF
    best_child::Node = n.children[1]
    for child in n.children
        if child.N^2 < floor(Int, kFP*child.pi_value*n.N)  # condition for forcing a playout on Katago paper. child.N > 0 replaced by floor(Int, ...)
            return child
        end
        cur = PUCT(child, false, true)
        if(cur > best_value)
            best_value = cur
            best_child = child
        end
    end
    return best_child
end
function player_of_tree(n::Node)
    if isnothing(n.parent)  # ROOT
        if length(n.children) == 0
            return 3-n.justPlayedPlayer
        else
            return n.children[1].justPlayedPlayer
        end
    else
        return player_of_tree(n.parent)
    end
end

function depth(n::Node) :: Int
    if isnothing(n.parent)
        return 1
    else
        return depth(n.parent)+1
    end
end
function depth_subtree(n::Node) :: Int
    if length(n.children) == 0
        return 0
    else
        return maximum([depth_subtree(c) for c in n.children])
    end
end

function is_finished_searching_exploitation_mode(n::Node, maxIter)
    # check if number 1 is reachable or not
    if length(n.children) == 1
        return n.N >= maxIter / 100
    end
    first, second = 0,0
    for c in n.children
        if c.N >= first
            first, second = c.N, first
        elseif c.N > second
            second = c.N
        end
    end
    remaining = maxIter - n.N
    return second + remaining < first
end
function is_finished_searching(n::Node, maxIter::Int) :: Bool
    #return n.N >= maxIter || n.is_terminal != NOT_TERMINAL
    return n.N >= maxIter
end

"""MCTS - AlphaZero environement : """

mutable struct AZAlgorithm  <: AbstractAlgorithm
    ROOT::Node
    nn::AbstractNN
    maxIteration::Int
    timeLimit::Float64
    AZAlgorithm(G::AbstractGame, nn::AbstractNN) = new(Node(G), nn, 1000, 0.0)
    AZAlgorithm(G::AbstractGame, nn::AbstractNN, maxIteration::Int, timeLimit::Float64) = new(Node(G), nn, maxIteration, timeLimit)
end



function iteration(AZ::AZAlgorithm, _G::Game.AbstractGame; cache=0)
    # performs a single iteration on ROOT
    G = deepcopy(_G)
    if (try TYPE_GAME <: Azul catch; false end)
        Random.seed!(G.core_rng, rand(Int))
    end

    # 1) select
    cur = AZ.ROOT
    while length(cur.children) > 0# && (isnothing(cur.parent) || !stopTreeExpansion(G))
        cur = select_child(cur)
        play!(G, cur.justPlayedMove)

        #=
        if G.playerToPlay == 0
            if length(cur.children) == 0 || 
                # Make new nodes
                for r in 1:PARALLEL_RANDOM_NODES
                    move = get_move_random_stuffs(G)
                    push!( cur.children, Node(move, 0, 1.0f0/PARALLEL_RANDOM_NODES, cur) )
                end
                cur = cur.children[1]
                play!(G, cur.justPlayedMove)

            elseif cur.N == CONDITION
                move = get_move_random_stuffs(G)
                cur = select_child(cur)
                play!(G, cur.justPlayedMove)
                push!( cur.children, Node(move, 0, 1.0f0/PARALLEL_RANDOM_NODES, cur) )
            else
                cur = select_child(cur)
                play!(G, cur.justPlayedMove)
            end
        end
        =#
    end

    # 2) expand
    winner::Int = -1
    #winExpectancy::Float32 = 0 # value of the node to be backpropagated (in range 0 - 1)
    winExpectancies = Float32[0 for p in 1:2] # values for each player to be backpropagated
    pointsExpectancies = [[0.0 for _ in 1:LENGTH_POINTS] for p in 1:2]    # delta points round, delta rawscore game, delta rows, col, colors
    
    if is_finished(G)
        winner = get_winner(G)
        if winner == 0  #DRAW
            winExpectancies = [0.5, 0.5]
        else
            winExpectancies = Float32[(p==winner) for p in 1:2]
        end

        if (try TYPE_GAME <: Azul catch; false end)
            delta = Game.get_delta_points(G)    #raw score, rows, cols, colors
            delta[1][1] -= _G.scores[1]  # remove actual score to get the score gain
            delta[2][1] -= _G.scores[2]
            for p in 1:2
                pointsExpectancies[p][ [1,3,4,5] ] = delta[p] - delta[3-p]
            end
        end

    else
        if G.playerToPlay == 0 # Make several parallel random nodes
            for r in 1:PARALLEL_RANDOM_NODES
                move = get_move_random_stuffs(G)
                push!( cur.children, Node(move, 0, 1.0f0/PARALLEL_RANDOM_NODES, cur) )
            end
            cur = cur.children[1]
            play!(G, cur.justPlayedMove)
        end

        GPUArrays.@cached cache begin
            output = NNet.predict(AZ.nn, G)
        end
        pi_predict = Flux.softmax(output[1])

        winExpectancies[G.playerToPlay]   =   output[2]
        winExpectancies[3-G.playerToPlay] = 1-output[2]

        if (try TYPE_GAME <: Azul catch; false end)
            #Warning delta points is [points won in round, points in remaining game, nrows, ncols, ncolors]
            if G.round == _G.round # didnt change round
                delta_points = output[3]
            else    # round changed : 
                p = G.playerToPlay
                delta_points = [G.scores[p]-G.scores[3-p]-_G.scores[p]+_G.scores[3-p], output[3][1] + output[3][2], output[3][3], output[3][4], output[3][5]]
            end
            pointsExpectancies[G.playerToPlay]   =  delta_points
            pointsExpectancies[3-G.playerToPlay] = -delta_points

        elseif TYPE_GAME <: AbstractPointsGame
            #v_predict = 0.5*output[NUMBER_ACTIONS+1] + 0.5*get_utility_score(G, output[NUMBER_ACTIONS+2])
            pointsExpectancies[G.playerToPlay]   =  0
            pointsExpectancies[3-G.playerToPlay] =  0
        end

        cur.v_initial = (TYPE_GAME <: AbstractPointsGame) ? [output[2], output[3]...] : [output[2]]   # therefore, for G.playerToPlay !!

        for move in all_moves(G)
            idx_output = get_idx_output(G, move)
            push!( cur.children, Node(move, G.playerToPlay, pi_predict[idx_output], cur) )
        end
        sum_pi = sum([c.pi_value for c in cur.children])
        for c in cur.children
            c.pi_value = (sum_pi == 0) ? 1.0 : c.pi_value / sum_pi
        end
    end

    # 4) backpropagate
    while !isnothing(cur)
        #@assert cur.justPlayedPlayer != 3
        cur.N += 1
        cur.w += winExpectancies[cur.justPlayedPlayer == 0 ? 1 : cur.justPlayedPlayer]          # If node is random, score as if it was to player 1
        cur.points += pointsExpectancies[cur.justPlayedPlayer == 0 ? 1 : cur.justPlayedPlayer]  # Same

        if !isnothing(cur.parent)
            L = 0.995
            cur.pi_value = L*cur.pi_value + (1.0-L) / length(cur.parent.children)
        end

        cur = cur.parent
    end
end

function add_dirichlet!(AZ::AZAlgorithm)
    pi_value = [n.pi_value for n in AZ.ROOT.children]
    pi_value = log.(pi_value) / 1.25   #removes softmax and scale by T
    Flux.softmax!(pi_value)

    n = length(AZ.ROOT.children)
    r = rand(Dirichlet(n, 10.0/n))
    for (c, new_pi, x) in zip(AZ.ROOT.children, pi_value, r)
#         c.pi_value = LEARNING_PARAMS["WEIGHT_DIRICHLET"] * x  + (1-LEARNING_PARAMS["WEIGHT_DIRICHLET"]) * pi_value
        #c.pi_value = new_pi + LEARNING_PARAMS["WEIGHT_DIRICHLET"] * (x - new_pi)
        c.pi_value = new_pi
        c.pi_dirichlet = x
    end

end



#=
function get_best_move(AZ::AZAlgorithm, G::AbstractGame, rng = Random.default_rng()) :: AbstractMove
    @assert false
    if AZ.ROOT.is_terminal == NOT_TERMINAL
        #Some moves lose (dont take), some move draw (50% wr), some moves are not terminal (x% wr), some moves have N=0 (dont take)
        considered = filter( n -> (n.is_terminal != TERMINAL_LOSS) && (n.N > 0), AZ.ROOT.children)
        if length(considered) == 0
            return rand(rng, AZ.ROOT.children).justPlayedMove
        else
            idx = findmax( n->n.N , considered)[2]
            return considered[idx].justPlayedMove
        end

    elseif AZ.ROOT.is_terminal == (G.playerToPlay == AZ.ROOT.justPlayedPlayer ? TERMINAL_WIN : TERMINAL_LOSS)
        # Find a winning move
        return rand(rng, filter(n -> n.is_terminal == TERMINAL_WIN, AZ.ROOT.children)).justPlayedMove

    elseif AZ.ROOT.is_terminal == TERMINAL_DRAW
        # Some moves draw.
        return rand(rng, filter(n -> n.is_terminal == TERMINAL_DRAW, AZ.ROOT.children)).justPlayedMove

    else
        #All moves lose. Take one with most N
        idx = findmax(n -> n.N, AZ.ROOT.children)[2]
        return AZ.ROOT.children[idx].justPlayedMove
    end
end

function get_move_temperature(AZ::AZAlgorithm, G::Game.AbstractGame, temperature::Float32, rng = Random.default_rng())
    if AZ.ROOT.is_terminal == NOT_TERMINAL && temperature > 0
        S = [(n.is_terminal == TERMINAL_LOSS ? 0 : n.N^1.0f0/temperature) for n in AZ.ROOT.children]
        if sum(S) > 0
            S /= sum(S)
            return StatsBase.sample(rng, AZ.ROOT.children, Weights(S)).justPlayedMove # library StatsBase
        else
            return rand(rng, AZ.ROOT.children).justPlayedMove
        end

    elseif AZ.ROOT.is_terminal == NOT_TERMINAL && temperature == 0
        idx = findmax( n-> n.is_terminal == TERMINAL_LOSS ? 0 : n.N , AZ.ROOT.children)[2]
        return AZ.ROOT.children[idx].justPlayedMove

    else
        return get_best_move(AZ, G, rng)
    end
end
    =#
#=
function get_move_temperature(AZ::AZAlgorithm, G::Game.Boop, temperature::Float32; rng=Random.default_rng(), forced_playouts=false)
    #Special for Boop because output for NN is different from moves (some moves duplicate outputs from NN)
    #Returns (move according to the temperature, pi and v)
    pi_MCTS::Vector{Float32} = zeros(NUMBER_ACTIONS)
    v_MCTS::Float32 = -1.0
    freq_MCTS::Vector{Float32} = zeros(length(AZ.ROOT.children))

    if AZ.ROOT.is_terminal == NOT_TERMINAL
        v_MCTS = 1-AZ.ROOT.w/AZ.ROOT.N
        for (i, child) in enumerate(AZ.ROOT.children) if child.is_terminal != TERMINAL_LOSS
            idx_output = get_idx_output(G, child.justPlayedMove)
            tmp = forced_playouts ? max(0, child.N - sqrt(kFP * child.pi_value * child.parent.N)) : child.N
            pi_MCTS[idx_output] = max(pi_MCTS[idx_output], tmp)
            freq_MCTS[i] = child.N
        end end

        if all(pi_MCTS .== 0.0)
            # happens if all children are 0 or loss
            # take all nodes even losses
            for (i, child) in enumerate(AZ.ROOT.children)
                idx_output = get_idx_output(G, child.justPlayedMove)
                pi_MCTS[idx_output] = child.N
                freq_MCTS[i] = child.N
            end
        end
        pi_MCTS /= sum(pi_MCTS)
        freq_MCTS /= sum(freq_MCTS)

        if temperature <= 0.01f0
            i_move = argmax(freq_MCTS)
        elseif temperature == 1.0f0
            i_move = rand(rng, Categorical(freq_MCTS))
        else
            P = freq_MCTS .^ (1.0f0/temperature)
            P /= sum(P)
            i_move = rand(rng, Categorical(P))
        end

    elseif AZ.ROOT.is_terminal == (G.playerToPlay == AZ.ROOT.justPlayedPlayer ? TERMINAL_LOSS : TERMINAL_WIN)     # all moves are loss : take a child at random
        v_MCTS = 0.0
        for (i, child) in enumerate(AZ.ROOT.children)
            idx_output = get_idx_output(G, child.justPlayedMove)
            pi_MCTS[ idx_output ] = 1
            freq_MCTS[i] = 1 / length(AZ.ROOT.children)
        end
        pi_MCTS /= sum(pi_MCTS)
        i_move = rand(rng, Categorical(freq_MCTS))

    elseif AZ.ROOT.is_terminal == TERMINAL_DRAW # all moves are draws : take a draw child at random
        v_MCTS = 0.5
        for (i, child) in enumerate(AZ.ROOT.children) if child.is_terminal == TERMINAL_DRAW
            idx_output = get_idx_output(G, child.justPlayedMove)
            pi_MCTS[ idx_output ] = 1
            freq_MCTS[i] = 1
        end end
        freq_MCTS /= sum(freq_MCTS)
        pi_MCTS /= sum(pi_MCTS)
        i_move = rand(rng, Categorical(freq_MCTS))

    else           # some moves are wins : take one at random
        v_MCTS = 1.0
        lowest_depth = INF
        for (i, child) in enumerate(AZ.ROOT.children) if child.is_terminal == TERMINAL_WIN
            if AZmodule.depth(child) == lowest_depth
                idx_output = get_idx_output(G, child.justPlayedMove)
                pi_MCTS[ idx_output ] = 1
                freq_MCTS[i] = 1
            elseif AZmodule.depth(child) < lowest_depth
                lowest_depth = AZmodule.depth(child)
                pi_MCTS *= 0
                freq_MCTS *= 0
                idx_output = get_idx_output(G, child.justPlayedMove)
                pi_MCTS[ idx_output ] = 1
                freq_MCTS[i] = 1
            end
        end end
        pi_MCTS /= sum(pi_MCTS)
        freq_MCTS /= sum(freq_MCTS)
        i_move = rand(rng, Categorical(freq_MCTS))
    end

    move = AZ.ROOT.children[i_move].justPlayedMove

    return move, pi_MCTS, v_MCTS
end=#

function get_best_move(AZ::AZAlgorithm, G::Game.AbstractGame, rng=Random.default_rng())
    bestN = -1
    candidates = []
    for c in AZ.ROOT.children
        if c.N > bestN
            candidates = [c]
            bestN = c.N
        elseif c.N == bestN
            push!(candidates, c)
        end
    end

    return rand(rng, candidates).justPlayedMove

end

function get_move_temperature(AZ::AZAlgorithm, G::AbstractGame, temperature::Float32; rng=Random.default_rng(), forced_playouts=false)
    pi_MCTS::Vector{Float32} = zeros(NUMBER_ACTIONS)
    v_MCTS::Float32 = -1.0
    points_MCTS = Float32[0 for i in 1:LENGTH_POINTS]

    #v_MCTS = (G.playerToPlay == AZ.ROOT.justPlayedPlayer ? AZ.ROOT.w/AZ.ROOT.N : 1-AZ.ROOT.w/AZ.ROOT.N)

    imax = findmax([c.N for c in AZ.ROOT.children])[2]
    v_MCTS = AZ.ROOT.children[imax].w / AZ.ROOT.children[imax].N
    points_MCTS = AZ.ROOT.children[imax].points / AZ.ROOT.children[imax].N

    for c in AZ.ROOT.children
        idx_output = get_idx_output(G, c.justPlayedMove)
        pi_MCTS[idx_output] = (GAME == "Boop") ? max(pi_MCTS[idx_output], forced_playouts ? max(0, c.N - sqrt(kFP * c.pi_value * c.parent.N)) : c.N) :
            forced_playouts ? max(0, c.N - sqrt(kFP * c.pi_value * c.parent.N)) : c.N

    end
    pi_MCTS /= sum(pi_MCTS)


    if temperature == 0.0f0
        idx_move = argmax(pi_MCTS)
    elseif temperature == 1.0f0
        idx_move = rand(rng, Categorical(pi_MCTS))
    else
        P = pi_MCTS .^ (1.0f0/temperature)
        P /= sum(P)
        idx_move = rand(rng, Categorical(P))
    end

    if GAME == "Boop"   # several moves have same id. Which one choose ?
        move = nothing
        bestN = -1
        for c in AZ.ROOT.children
            if get_idx_output(G, c.justPlayedMove) == idx_move && c.N > bestN
                bestN = c.N
                move = c.justPlayedMove
            end
        end
        @assert bestN != -1
    else
        move = get_move_from_idx_move(G, idx_move)
    end

    return move, pi_MCTS, v_MCTS, points_MCTS
end


end # AZmodule


using .AZmodule
using StatsBase
using GPUArrays
using DataFrames

function _print(node::AZmodule.Node, G::AbstractGame; indent=0, depth_child=0, top=100)
    tab = [(c.N, c) for c in node.children]
    sort!(tab, by = c -> c[1], rev = true)

    #df = DataFrame(
    #    Move = [get_printable_move(G, c.justPlayedMove) for (N,c) in tab],
    #    Score = [round(Int, 100*c.w/max(1,c.N)) for (N,c) in tab],
    #    PiNN = [round(1000*c.pi_value)/10 for (N,c) in tab],
    #    vNN = [c.v_initial for (N,c) in tab]
    #)
    #print(df)
    #display(df)

    function _length(move, G)
        s = get_printable_move(G, move)
        if (try TYPE_GAME <: Azul catch; return false end)
            if G.playerToPlay == 0
                return 0
            elseif GAME == "Azul"
                tiles = (move.kind == 2 #= ACTION_PLACE =#) ? sum(G.buffer) : G.factory[move.line, move.color]
            else
                tmp = move.code
                color = mod(tmp-1, 5)+1
                tmp = div(tmp-color,5)+1
                to = mod(tmp-1, 6)+1
                from = div(tmp-to, 6)+1
                tiles = G.factory[from, color]
            end
            return length(s) - 10*tiles
        else
            return length(s)
        end
    end
    text_small_float(x) = abs(x) < 0.1 ? "0   " : string(x)[1:min(4, length(string(x)))]
    text_array_float(arr) = join([text_small_float(x) for x in arr], " ")
    text_percentage(f) = f==0 ? "0%   " : string(div(round(Int,f*1000), 10))*"."*string(mod(round(Int,f*1000), 10))*"%"
    maxsize = maximum( [_length(c.justPlayedMove, G) for (N,c) in tab] )

    for (N, c) in tab[1:min(top, length(tab))]
        (c.N > 0) || break
        string_move = get_printable_move(G, c.justPlayedMove) * join( [" " for i in 1:(maxsize+1-_length(c.justPlayedMove, G))] )
        for i in 1:indent
            print("\t")
        end
        
        if TYPE_GAME <: AbstractPointsGame
            println(string_move, " ",  text_percentage(c.w/max(1, c.N)), "\tN=", N, "\tpi= ", text_percentage(c.pi_value), "\tpoints:", (GAME=="Azul") ? text_small_float([1,1,2,5,10]'*c.points/max(1,c.N))*" " : "", text_percentage(points_to_utility(G, c.points/max(1,c.N))), "\tdetail: ", text_array_float(c.points/max(1,c.N)), "\tinit=", text_array_float(c.v_initial)) #, "\tpi_nn = ", round(1000*c.pi_value)/10, "\tw_init= ", c.v_initial)
        else
            println(string_move, " ",  text_percentage(c.w/max(1, c.N)), "\tN=", N, "\tpi= ", text_percentage(c.pi_value), "\tinit=", text_array_float(c.v_initial)) #, "\tpi_nn = ", round(1000*c.pi_value)/10, "\tw_init= ", c.v_initial)
        end
        if depth_child>0 && length(c.children) > 0
            G2 = deepcopy(G)
            play!(G2, c.justPlayedMove)
            _print(c, G2; indent=indent+1, depth_child=depth_child-1, top= (top>3) ? 3 : max(1, top-1))
        end
    end
end

function Base.print(AZ::AZAlgorithm, G::AbstractGame ; depth_child=0, top=100)
    print("Number of iterations : ", AZ.ROOT.N)
    w = AZ.ROOT.justPlayedPlayer == G.playerToPlay ? AZ.ROOT.w / max(1,AZ.ROOT.N) : 1-AZ.ROOT.w / max(1,AZ.ROOT.N)
    println(" w = ", floor(Int, 100*w), "%\tinit: ", AZ.ROOT.v_initial)
    _print(AZ.ROOT, G; indent=0, depth_child=depth_child, top=top)
end

function Algorithm.get_move(AZ::AZAlgorithm, G::AbstractGame, temperature::Float32 = 0.0f0) :: AbstractMove
    start = 0   # trick to avoid compilation time that can be huge
    cache = GPUArrays.AllocCache()
    @assert AZ.maxIteration + AZ.timeLimit > 0 string(AZ.maxIteration)*" "*string(AZ.timeLimit)
    maxIteration = AZ.maxIteration
    timeLimit = AZ.timeLimit
    iter = 0
    while (AZ.maxIteration == 0 || (AZ.maxIteration > 0 && AZ.ROOT.N < maxIteration)) &&
            (AZ.timeLimit == 0 || start == 0 || (AZ.timeLimit > 0 && time() - start < timeLimit))
        AZmodule.iteration(AZ, G; cache=cache)
        iter += 1
        maxIteration = length(AZ.ROOT.children) == 1 ? div(AZ.maxIteration, 10) : AZ.maxIteration
        timeLimit = length(AZ.ROOT.children) == 1 ? div(AZ.timeLimit, 10) : AZ.timeLimit
        if start == 0
            start = time()
        end
    end

    if temperature == 0.0f0
        return AZmodule.get_best_move(AZ, G)
    else
        move, = AZmodule.get_move_temperature(AZ, G, temperature)
        return move
    end
end

function Algorithm.get_move(AZ::AZAlgorithm, G::AbstractRandomGame, temperature::Float32 = 0.0f0 ) :: AbstractMove

    moves = all_moves(G)
    if length(moves) == 1
        AZ.ROOT.w, AZ.ROOT.N = 0.5, 1
        return moves[1]
    end
    AZ.ROOT = AZmodule.Node(G)
    AZmodule.iteration(AZ, G)

    for U in 1:UNIVERSES
        Random.seed!(G.core_rng, rand(Int))
        AZtmp = AZAlgorithm(G, AZ.nn, AZ.maxIteration, AZ.timeLimit)
        start = 0   # trick to avoid compilation time that can be huge
        while (((AZtmp.maxIteration > 0 && AZtmp.ROOT.N < AZtmp.maxIteration) || (start == 0) || (AZtmp.timeLimit > 0 && time() - start < AZtmp.timeLimit))) && !AZmodule.all_children_finished(AZtmp.ROOT)
            AZmodule.iteration(AZtmp, G)
            if start == 0
                start = time()
            end
        end

        if AZtmp.ROOT.is_terminal == NOT_TERMINAL
            AZ.ROOT.N += 1
            AZ.ROOT.w += AZtmp.ROOT.w / AZtmp.ROOT.N
            total = max(1,sum([c.N for c in AZtmp.ROOT.children]))
            for (i, c) in enumerate(AZtmp.ROOT.children)
                AZ.ROOT.children[i].w += (c.is_terminal == TERMINAL_LOSS) ? 0 : ((c.is_terminal == TERMINAL_DRAW) ? 0.5 : c.w/max(1,c.N))
                AZ.ROOT.children[i].N += (c.is_terminal == TERMINAL_LOSS) ? 0 : round(Int,100*c.N/total)
            end

        elseif AZtmp.ROOT.is_terminal == TERMINAL_WIN  # All moves are 0
            AZ.ROOT.w += 0
            AZ.ROOT.N += 1
            total = length(AZtmp.ROOT.children)
            for (i, c) in enumerate(AZtmp.ROOT.children)
                AZ.ROOT.children[i].w += 0
                AZ.ROOT.children[i].N += round(Int, 100/total)
            end

        elseif AZtmp.ROOT.is_terminal == TERMINAL_DRAW # Some draw moves some are losses
            AZ.ROOT.w += 0.5
            AZ.ROOT.N += 1
            total = sum([c.is_terminal == TERMINAL_DRAW for c in AZtmp.ROOT.children])
            for (i, c) in enumerate(AZtmp.ROOT.children)
                AZ.ROOT.children[i].w += (c.is_terminal == TERMINAL_DRAW) ? 0.5 : 0.0
                AZ.ROOT.children[i].N += (c.is_terminal == TERMINAL_DRAW) ? round(Int, 100/total) : 0
            end

        elseif AZtmp.ROOT.is_terminal == TERMINAL_LOSS  # one win
            AZ.ROOT.w += 1.0
            AZ.ROOT.N += 1
            total = sum([c.is_terminal == TERMINAL_WIN for c in AZtmp.ROOT.children])
            for (i, c) in enumerate(AZtmp.ROOT.children)
                AZ.ROOT.children[i].w +=  [c.w/max(1,c.N), 0.0, 0.5, 1.0][c.is_terminal+1]
                AZ.ROOT.children[i].N += (c.is_terminal == TERMINAL_WIN) ? round(Int, 100/total) : 0    # Not incredible... Not winning move are 0 but in other universes they can be good
            end

        end


        if U == 1
            println("ici U=1")
            print(AZtmp, G)
        end
    end



    for c in AZ.ROOT.children
        c.N = round(Int, c.N / UNIVERSES)
        c.w = c.w / UNIVERSES * c.N
    end

    Random.seed!(G.core_rng, rand(Int))
    println("done ", AZ.ROOT.w)

    if temperature == 0.0f0
        idx = findmax([c.N for c in AZ.ROOT.children])[2]
        return AZ.ROOT.children[idx].justPlayedMove
    else
        error("pas de temperature")
        maxi = maximum([c.N for c in AZ.ROOT.children])
        S = [c.N/maxi for c in AZ.ROOT.children] .^ (1.0f0/temperature)
        S /= sum(S)
        return StatsBase.sample(AZ.ROOT.children, Weights(S)).justPlayedMove
    end



end


function Algorithm.keep_subtree!(AZ::AZAlgorithm, move::AbstractMove)
    for child in AZ.ROOT.children
        if child.justPlayedMove == move
            if child.N > 0 && length(child.children) == 0
                break
            end
            AZ.ROOT = child
            AZ.ROOT.parent = nothing
            return
        end
    end
    AZ.ROOT = AZmodule.Node(move, AZ.ROOT.justPlayedPlayer)
end

function reset(AZ::AZAlgorithm, move::AbstractMove)
    AZ.ROOT = Node(move, AZ.ROOT.justPlayedPlayer)
end

function reset!(AZ::AZAlgorithm, move::AbstractMove)
    AZ.ROOT = Node(move, AZ.ROOT.justPlayedPlayer)
end