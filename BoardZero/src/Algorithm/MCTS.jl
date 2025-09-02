module MCTSmodule

export MCTSAlgorithm

using ..Game
using ..Algorithm
using Random

# MCTS creates a tree with many "Node"
# The top one being called "ROOT"

include("../params.jl")

#const CUCB = 0.3

mutable struct Node
    justPlayedMove::AbstractMove
    justPlayedPlayer::Int # 1 or 2
    w::Float64
    N::Int
    children::Vector{Node}
    parent::Union{Nothing,Node}
    is_terminal::Int # values : 0 if not terminal, or TERMINAL_WIN, or TERMINAL_LOSS, or TERMINAL_DRAW

    Node(move::AbstractMove, playerToPlayAfterMove::Int, parent) = new(move, 3-playerToPlayAfterMove, 0.0, 0, Vector{Node}(), parent, 0)
    Node(G::AbstractGame) = new(get_null_move(G), 3-G.playerToPlay, 0.0, 0, Vector{Node}(), nothing, 0)
end


function UCB(n::Node) :: Float64
#     if n.is_terminal == TERMINAL_LOSS || n.is_terminal == TERMINAL_WIN   # ne pas le prendre !!
#         return -INF
#     end
    if n.N == 0
        return INF
    else
        return n.w / n.N    +    CUCB * sqrt(log(n.parent.N) / n.N)
    end
end

function select_child(n::Node)  :: Node
    # Select child with highest UCB
    best_value = -INF
    best_child::Node = n.children[1]
    for child in n.children
        cur = UCB(child)
        if(cur > best_value)
            best_value = cur
            best_child = child
        end
    end
    return best_child
end

function get_player_tree(n::Node) :: Int
    return isnothing(n.parent) ? 3-n.justPlayedPlayer : get_player_tree(n.parent)
end



function backpropagate_status(n::Node)
    # Node n has just been marked as terminal
    if isnothing(n.parent)
        return
    end

    changingPlayer = n.justPlayedPlayer != n.parent.justPlayedPlayer

    if n.is_terminal == TERMINAL_WIN    # automatic loss for parent
        n.parent.is_terminal = changingPlayer ? TERMINAL_LOSS : TERMINAL_WIN
        backpropagate_status(n.parent)
    else    # count all draw and loss of children. ALL loss = WIN ; ALL draw = draw ; one not terminal = not terminal
        has_draw = false
        for child in n.parent.children
            if child.is_terminal == NOT_TERMINAL
                return nothing
            elseif child.is_terminal == TERMINAL_DRAW
                has_draw = true
            end
        end
        if has_draw
            n.parent.is_terminal = TERMINAL_DRAW
        else
            n.parent.is_terminal = changingPlayer ? TERMINAL_WIN : TERMINAL_LOSS
        end
        backpropagate_status(n.parent)
    end
end


mutable struct MCTSAlgorithm <: AbstractAlgorithm
    ROOT::Node
    maxIteration::Int       # 0 for infinity
    timeLimit::Float64      # in sec / 0 for infinity
    MCTSAlgorithm(G::AbstractGame) = new(Node(G), 10000, 0)
    MCTSAlgorithm(G::AbstractGame, max_iteration::Int, ms_time_limit::Float64) = new(Node(G), max_iteration, ms_time_limit)
end

function iteration(MCTS::MCTSAlgorithm, _G::AbstractGame, rng::Random.AbstractRNG)  :: Nothing
    # Performs a single iteration on ROOT
    G = deepcopy(_G)

    # 1) select
    cur = MCTS.ROOT
    while cur.is_terminal == NOT_TERMINAL && length(cur.children) > 0
        cur = select_child(cur)
        play!(G, cur.justPlayedMove)
    end

    # 2) expand
    winner::Int = -1    # get winner of current game : 1, 2 or 0(draw)
    if cur.is_terminal > 0
        if cur.is_terminal == TERMINAL_WIN
            winner = cur.just_played_player
        elseif cur.is_terminal == TERMINAL_DRAW
            winner = 0
        else
            winner = 3-cur.justPlayedPlayer
        end

    elseif is_finished(G)
        winner = get_winner(G)
        if winner == 0
            cur.is_terminal = TERMINAL_DRAW
        elseif winner == cur.justPlayedPlayer
            cur.is_terminal = TERMINAL_WIN
        else
            cur.is_terminal = TERMINAL_LOSS
        end
        backpropagate_status(cur)

    else
        if G.playerToPlay == 0 # Make several parallel random nodes
            for r in 1:PARALLEL_RANDOM_NODES
                move = get_move_random_stuffs(G)
                push!( cur.children, Node(move, 0, cur) )
            end
            cur = cur.children[1]
            play!(G, cur.justPlayedMove)
        end

        cur.children = [Node(move, 3-G.playerToPlay, cur) for move in Random.shuffle(rng, all_moves(G))]
#         for move in Random.shuffle(rng, all_moves(G))
#             #push!( cur.children, Node(move, G.playerToPlay, 0, 0, Vector{Node}(), cur, 0) )
#             push!( cur.children, Node(move, 3-G.playerToPlay, cur) )
#         end

        # 3) rollout
        if length(cur.children) == 0
            println("pas de chidlren !")
            print(G)
        end
        cur = rand(rng, cur.children)
        play!(G, cur.justPlayedMove)
        play_random!(G, rng)
        winner = get_winner(G)
    end

    # 4) backpropagate
    while !isnothing(cur)
        cur.N += 1
        if cur.justPlayedPlayer == winner
            cur.w += 1.0
        elseif winner == 0  #draw
            cur.w += 0.5
        end
        cur = cur.parent
    end
    return nothing
end



end # end module


using .MCTSmodule

function _print(node::MCTSmodule.Node, G::AbstractGame; indent=0, depth_child=0, top=100)
    sort!(node.children, by = n -> [n.w / (n.N+1), 0, 0.5, 1.0][n.is_terminal+1], rev = true)
    for c in node.children[1:min(top, length(node.children))]
        string_move = get_printable_move(G, c.justPlayedMove)
        for i in 1:indent
            print("\t")
        end
        if c.is_terminal > 0
            println(string_move, "\tN= ", c.N, "\t", ["LOSS", "DRAW", "WIN"][c.is_terminal])
        else
            println(string_move, "\tscore : ", c.N==0 ? 50 : floor(Int, 100*c.w/c.N), "%\tN=", c.N)
        end
        if depth_child>0 && length(c.children) > 0
            G2 = deepcopy(G)
            play!(G2, c.justPlayedMove)
            _print(c, G2; indent=indent+1, depth_child=depth_child-1, top= (top>3) ? 3 : max(1, top-1))
        end
    end
end

function Base.print(MCTS::MCTSAlgorithm, G::AbstractGame; depth_child=0, top=100)
    print("Number of iterations : ", MCTS.ROOT.N)
    if MCTS.ROOT.is_terminal > 0
        println([" WIN ", " DRAW ", " LOSS "][MCTS.ROOT.is_terminal])
    else
        println(" w = ", floor(Int, 100-100*MCTS.ROOT.w/(MCTS.ROOT.N+1)), "%")
    end

    _print(MCTS.ROOT, G; indent=0, depth_child=depth_child, top=top)
end

function Algorithm.get_move(MCTS::MCTSAlgorithm, G::AbstractGame, rng = Random.default_rng()) :: AbstractMove
    # iterate in the tree
    start = time()
    while ((MCTS.maxIteration > 0 && MCTS.ROOT.N < MCTS.maxIteration) || (MCTS.timeLimit > 0 && time() - start < MCTS.timeLimit)) && MCTS.ROOT.is_terminal == NOT_TERMINAL
        MCTSmodule.iteration(MCTS, G, rng)
    end

    # find best move
    if MCTS.ROOT.is_terminal == NOT_TERMINAL
        #Some moves lose (dont take), some move draw (50% wr), some moves are not terminal (x% wr), some moves have N=0 (dont take)
        considered = filter( n -> (n.is_terminal != TERMINAL_LOSS) && (n.N > 0), MCTS.ROOT.children)
        if length(considered) == 0
            return rand(rng, MCTS.ROOT.children).justPlayedMove
        else
            idx = findmax( n-> n.is_terminal == 0 ? n.w / n.N : 0.5 , considered)[2]
            return considered[idx].justPlayedMove
        end

    elseif MCTS.ROOT.is_terminal == (G.playerToPlay == MCTS.ROOT.justPlayedPlayer ? TERMINAL_WIN : TERMINAL_LOSS) # Win for playerToPlay
        # Find a winning move
        return rand(rng, filter(n -> n.is_terminal == TERMINAL_WIN, MCTS.ROOT.children)).justPlayedMove

    else
        # All moves lose or all moves draw. In both cases get move with most N
        idx = findmax(n -> n.N, MCTS.ROOT.children)[2]
        return MCTS.ROOT.children[idx].justPlayedMove
    end
end

function Algorithm.get_move(MCTS::MCTSAlgorithm, G::AbstractRandomGame, rng = Random.default_rng()) :: AbstractMove
    # iterate in the tree
    moves = all_moves(G)
    if length(moves) == 1
        return moves[1]
    end
    v_mean, pi_mean, all_v = 0, Dict{Any, Float64}([Game.move_to_string(G, m) => 0 for m in moves]), Dict{Any, Float64}([Game.move_to_string(G, m) => 0 for m in moves])
    for U in 1:UNIVERSES
        seed = rand(Int)
        MCTS = MCTSAlgorithm(G, MCTS.maxIteration, MCTS.timeLimit)
        start = time()
        while ((MCTS.maxIteration > 0 && MCTS.ROOT.N < MCTS.maxIteration) || (MCTS.timeLimit > 0 && time() - start < MCTS.timeLimit)) && MCTS.ROOT.is_terminal == NOT_TERMINAL
            Random.seed!(G.core_rng, seed)
            MCTSmodule.iteration(MCTS, G, rng)
        end

        if MCTS.ROOT.is_terminal == NOT_TERMINAL
            @assert G.playerToPlay != MCTS.ROOT.justPlayedPlayer
            v_cur = 1-MCTS.ROOT.w/MCTS.ROOT.N
            pi_cur = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => (c.is_terminal == TERMINAL_LOSS) ? 0 : c.N for c in MCTS.ROOT.children])
            vs = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => (c.N == 0 || c.is_terminal == TERMINAL_LOSS) ? 0 : c.w / c.N for c in MCTS.ROOT.children])


        elseif MCTS.ROOT.is_terminal == TERMINAL_WIN  # all moves are loss
            v_cur = 0
            pi_cur = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => 0 for c in MCTS.ROOT.children])
            vs = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => 0 for c in MCTS.ROOT.children])

        elseif MCTS.ROOT.is_terminal == TERMINAL_DRAW   # all moves are loss or draw
            v_cur = 0.5
            pi_cur = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => (c.is_terminal == TERMINAL_DRAW) for c in MCTS.ROOT.children])
            vs = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => 0.5*(c.is_terminal == TERMINAL_DRAW) for c in MCTS.ROOT.children])

        else   # a move is a win
            v_cur = 1.0
            pi_cur = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => c.is_terminal == TERMINAL_WIN for c in MCTS.ROOT.children])
            vs = Dict{Any, Float64}([Game.move_to_string(G, c.justPlayedMove) => (c.is_terminal == TERMINAL_WIN) for c in MCTS.ROOT.children])

        end

        total = sum([pi_cur[m] for m in keys(pi_cur)])
        if total != 0
            for m in keys(pi_cur)
                pi_cur[m] /= total
            end
        end

#         print("Universe number ", U, "\tv = ", v_cur)
#         print(MCTS, G)

        v_mean += v_cur
        pi_mean = Dict{Any, Float64}([m => pi_mean[m] + pi_cur[m] for m in keys(pi_cur)])
        all_v = Dict{Any, Float64}([m => all_v[m] + vs[m] for m in keys(vs)])
    end

    v_mean /= UNIVERSES

    move = Game.string_to_move(G, findmax(pi_mean)[2])

    Random.seed!(G.core_rng, rand(Int))

    # TO PRINT
    MCTS = MCTSAlgorithm(G, MCTS.maxIteration, MCTS.timeLimit)
    MCTSmodule.iteration(MCTS, G, rng)
    MCTS.ROOT.w = 1-v_mean
    for (i, m) in enumerate(moves)
        name_move = Game.move_to_string(G, MCTS.ROOT.children[i].justPlayedMove)
        MCTS.ROOT.children[i].N = round(Int, MCTS.maxIteration * pi_mean[ name_move ] / UNIVERSES)
        MCTS.ROOT.children[i].w = all_v[ name_move ] / UNIVERSES * MCTS.ROOT.children[i].N
    end

    return move

end


function Algorithm.keep_subtree!(MCTS::MCTSAlgorithm, move::AbstractMove)
    for child in MCTS.ROOT.children
        if child.justPlayedMove == move
            MCTS.ROOT = child
            MCTS.ROOT.parent = nothing
            return
        end
    end
    MCTS.ROOT = MCTSmodule.Node(move, MCTS.ROOT.justPlayedPlayer, nothing)
end
