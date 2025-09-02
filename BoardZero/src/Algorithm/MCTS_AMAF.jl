module AMAFmodule

export AMAFAlgorithm

using ..Game
using ..Algorithm
using Random, DataStructures

# MCTS creates a tree with many "Node"
# The top one being called "ROOT"

include("../params.jl")

mutable struct Node
    justPlayedMove::AbstractMove
    justPlayedPlayer::Int # 1 or 2
    w::Float64
    N::Int
    U::Float64
    NAMAF::Int
    children::Vector{Node}
    parent::Union{Nothing,Node}
    is_terminal::Int # values : 0 if not terminal, or TERMINAL_WIN, or TERMINAL_LOSS, or TERMINAL_DRAW

    Node(move::AbstractMove, playerToPlayAfterMove::Int, parent) = new(move, 3-playerToPlayAfterMove, 0.0, 0, 0.0, 0, Vector{Node}(), parent, 0)
    Node(G::AbstractGame) = new(get_null_move(G), 3-G.playerToPlay, 0.0, 0, 0.0, 0, Vector{Node}(), nothing, 0)
end


function UCB(n::Node) :: Float64
    CRAVE = (get_player_tree(n) == 1 ? 200 : 200)
    if n.is_terminal == TERMINAL_LOSS   # don't take it
        return -INF
    elseif n.N + n.NAMAF == 0
        return INF
    elseif n.N == 0
        return INF + n.U / n.NAMAF
    elseif n.N > CRAVE
        return n.w / n.N    +    CUCB * sqrt(log(n.parent.N) / n.N)
    else
        alpha::Float64 = (CRAVE - n.N) / CRAVE
        #alpha::Float64 = sqrt(CRAVE / (3*n.N + CRAVE)) #CRAVE = 10
        return (1.0-alpha) * n.w / n.N + alpha * n.U / n.NAMAF  +  CUCB * sqrt(log(n.parent.N) / n.N)
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
    if isnothing(n.parent)
        return n.justPlayedPlayer
    else
        return 3-get_player_tree(n.parent)
    end
end

function backpropagate_status(n::Node)  :: Nothing
    # Node n has just been marked as terminal
    if isnothing(n.parent)
        return nothing
    elseif n.is_terminal == TERMINAL_WIN    # automatic loss for parent
        n.parent.is_terminal = TERMINAL_LOSS
        return backpropagate_status(n.parent)
    else    # count all draw and loss of children. ALL loss = WIN ; ALL draw = draw ; one not terminal = not terminal
        has_draw = false
        for child in n.parent.children
            if child.is_terminal == 0
                return nothing
            elseif child.is_terminal == TERMINAL_DRAW
                has_draw = true
            end
        end
        if has_draw
            n.parent.is_terminal = TERMINAL_DRAW
        else
            n.parent.is_terminal = TERMINAL_WIN
        end
        return backpropagate_status(n.parent)
    end
end


mutable struct AMAFAlgorithm <: AbstractAlgorithm
    ROOT::Node
    maxIteration::Int       # 0 for infinity
    timeLimit::Float64      # in sec / 0 for infinity
    AMAFAlgorithm(G::AbstractGame) = new(Node(G), 10000, 0.0)
    AMAFAlgorithm(G::AbstractGame, max_iteration::Int, ms_time_limit::Float64) = new(Node(G), max_iteration, ms_time_limit)
end


function iteration(MCTS::AMAFAlgorithm, _G::AbstractGame, rng::Random.AbstractRNG)  :: Nothing
    # Performs a single iteration on ROOT
    G = deepcopy(_G)

    # 1) select
    cur = MCTS.ROOT
    while cur.is_terminal == 0 && length(cur.children) > 0
        cur = select_child(cur)
        play!(G, cur.justPlayedMove)
    end

    # 2) expand
    winner::Int = -1    # get winner of current game : 1, 2 or 0(draw)
    didRollout::Bool = false    # to know if we did rollout or not, for AMAF
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
        cur.children = [Node(move, 3-G.playerToPlay, cur) for move in Random.shuffle(rng, all_moves(G))]

        # 3) rollout : play random move and store it
        cur = rand(rng, cur.children)
        play!(G, cur.justPlayedMove)
        played_moves = Vector{ SortedSet{AbstractMove} }([SortedSet{AbstractMove}(), SortedSet{AbstractMove}()])
        while !is_finished(G)
            move = random_move(G)
            push!(played_moves[G.playerToPlay] , move )
            play!(G, move)
        end
        winner = get_winner(G)
        didRollout = true
    end

    # 4) backpropagate status of the game for UCT and AMAF
    while !isnothing(cur)
        # UCT :
        cur.N += 1
        if cur.justPlayedPlayer == winner
            cur.w += 1.0
        elseif winner == 0  #draw
            cur.w += 0.5
        end

        # AMAF :
        if didRollout
            for child in cur.children
                if child.justPlayedMove in played_moves[child.justPlayedPlayer]
                    child.NAMAF += 1
                    child.U += (winner == 0) ? 0.5 : ((winner == child.justPlayedPlayer) ? 1.0 : 0.0 )
                end
            end

            cur.NAMAF += 1
            cur.U += (winner == 0) ? 0.5 : ((winner == cur.justPlayedPlayer) ? 1.0 : 0.0 )
        end

        cur = cur.parent
    end
    return nothing
end




end # end module


using .AMAFmodule

function Base.print(MCTS::AMAFAlgorithm, G::AbstractGame) :: Nothing
    print("Number of iterations : ", MCTS.ROOT.N)
    if MCTS.ROOT.is_terminal > 0
        println([" WIN ", " DRAW ", " LOSS "][MCTS.ROOT.is_terminal])
    else
        println(" w = ", floor(Int, 100-100*MCTS.ROOT.w/(MCTS.ROOT.N+1)), "%")
    end

    sort!(MCTS.ROOT.children, by = n -> [n.w / (n.N+1), 0, 0.5, 1.0][n.is_terminal+1], rev = true)

    for n in MCTS.ROOT.children[1:min(30, length(MCTS.ROOT.children))]
        string_move = get_printable_move(G, n.justPlayedMove)
        if n.is_terminal > 0
            println(string_move, "\tN= ", n.N, "\t", ["LOSS", "DRAW", "WIN"][n.is_terminal], "\t(U= ", floor(Int, 100*n.U/n.NAMAF), "%\tNAMAF= ", n.NAMAF, ")")
        else
            println(string_move, "\tscore : ", n.N==0 ? 50 : floor(Int, 100*n.w/n.N), "%\tN=", n.N, "\t(U= ", floor(Int, 100*n.U/ n.NAMAF), "%\tNAMAF= ", n.NAMAF, ")")
        end
    end
    return nothing
end

function Algorithm.get_move(MCTS::AMAFAlgorithm, G::AbstractGame, rng = Random.default_rng()) :: AbstractMove
    # iterate in the tree
    start = time()
    while ((MCTS.maxIteration > 0 && MCTS.ROOT.N < MCTS.maxIteration) || (MCTS.timeLimit > 0 && time() - start < MCTS.timeLimit)) && MCTS.ROOT.is_terminal == 0
        AMAFmodule.iteration(MCTS, G, rng)
    end

    # find best move
    if MCTS.ROOT.is_terminal == 0
        #Some moves lose (dont take), some move draw (50% wr), some moves are not terminal (x% wr), some moves have N=0 (dont take)
        considered = filter( n -> (n.is_terminal != TERMINAL_LOSS) && (n.N > 0), MCTS.ROOT.children)
        if length(considered) == 0
            return rand(MCTS.ROOT.children).justPlayedMove
        else
            idx = findmax( n-> n.is_terminal == 0 ? n.w / n.N : 0.5 , considered)[2]
            return considered[idx].justPlayedMove
        end

    elseif MCTS.ROOT.is_terminal == TERMINAL_LOSS
        # Find a winning move
        return rand(filter(n -> n.is_terminal == TERMINAL_WIN, MCTS.ROOT.children)).justPlayedMove

    else
        # All moves lose or all moves draw. In both cases get move with most N
        idx = findmax(n -> n.N, MCTS.ROOT.children)[2]
        return MCTS.ROOT.children[idx].justPlayedMove

    end
end


function Algorithm.keep_subtree!(MCTS::AMAFAlgorithm, move::AbstractMove) :: Nothing
    for child in MCTS.ROOT.children
        if child.justPlayedMove == move
            MCTS.ROOT = child
            MCTS.ROOT.parent = nothing
            return nothing
        end
    end
    MCTS.ROOT = AMAFmodule.Node(move, MCTS.ROOT.justPlayedPlayer, nothing)
    return nothing
end
