module ParallelAZ

using ..Algorithm, ..AZmodule
using ..Game, ..NNet

using Distributions: Categorical, Dirichlet
using Flux
using Random

include("../../params.jl")

export before_nn_eval!, after_nn_eval!

function before_nn_eval!(AZ::AZAlgorithm, _G::AbstractGame, G::Game.AbstractGame; forced_playouts::Bool=false) :: AZmodule.Node
    #play game G (to be stored) from _G, select a node (to be returned) and store the corresponding position
    load_from(_G, G)

    cur = AZ.ROOT
    if forced_playouts && length(AZ.ROOT.children) > 0
        cur = AZmodule.select_child_forced_playouts(AZ.ROOT)
        play!(G, cur.justPlayedMove)
    end
    while length(cur.children) > 0
        cur = AZmodule.select_child(cur)
        play!(G, cur.justPlayedMove)
    end

    if G.playerToPlay == 0 && !is_finished(G)
        # Make several parallel random nodes
        for r in 1:PARALLEL_RANDOM_NODES
            move = get_move_random_stuffs(G)
            if GAME == "Azul"   # We want all games to be the same
                move.seed = r + round(Int, time()/3600)*1000 + _G.round*10000000
            end
            push!( cur.children, AZmodule.Node(move, 0, 1.0f0/PARALLEL_RANDOM_NODES, cur) )
        end
        cur = cur.children[1]
        play!(G, cur.justPlayedMove)
    end


    return cur
end


function after_nn_eval!(cur::AZmodule.Node, G::AbstractGame, Gbefore::AbstractGame, output_nn, i_output::Int; rng=Random.default_rng(), add_dirichlet::Bool=false)
    #from output of NN (at position i_output), get pi and v to finish the MCTS iteration
    winExpectancies = Float32[0.5, 0.5]
    pointsExpectancies = [Float32[0 for i in 1:LENGTH_POINTS] for p in 1:2]

    if is_finished(G)
        winner = get_winner(G)
        if winner == 0
            winExpectancies = [0.5, 0.5]
        else
            winExpectancies = Float32[(p==winner) for p in 1:2]
        end

        if (try TYPE_GAME <: Azul catch; false end)
            delta = Game.get_delta_points(G)    #raw score, rows, cols, colors
            delta[1][1] -= Gbefore.scores[1]  # remove actual score difference to get the score gain
            delta[2][1] -= Gbefore.scores[2]
            for p in 1:2
                pointsExpectancies[p][ [1,3,4,5] ] = delta[p] - delta[3-p]
            end
        end

    else

        pi_predict::Vector{Float32} = output_nn[1][:, i_output]

        winExpectancies[G.playerToPlay]   =   output_nn[2][i_output]
        winExpectancies[3-G.playerToPlay] = 1-output_nn[2][i_output]

        if (try TYPE_GAME <: Azul catch; false end)
            if G.round == Gbefore.round
                delta_points =  output_nn[3][:, i_output]
            else
                p = G.playerToPlay
                delta_points = [G.scores[p]-G.scores[3-p]-Gbefore.scores[p]+Gbefore.scores[3-p], output_nn[3][1, i_output] + output_nn[3][2, i_output], output_nn[3][3, i_output], output_nn[3][4, i_output], output_nn[3][5, i_output]]
            end
            pointsExpectancies[G.playerToPlay]   =  delta_points
            pointsExpectancies[3-G.playerToPlay] = -delta_points

        elseif TYPE_GAME <: AbstractPointsGame
            pointsExpectancies[G.playerToPlay]   =  0
            pointsExpectancies[3-G.playerToPlay] =  0

        end

        
        cur.v_initial = (TYPE_GAME <: AbstractPointsGame) ? [output_nn[2][i_output], output_nn[3][i_output]...] : [output_nn[2][i_output]] 

        moves = all_moves(G)
        if add_dirichlet
            dirichlet::Vector{Float32} = rand(rng, Dirichlet(length(moves), 10.0 / length(moves)))
            pi_predict = Flux.softmax(pi_predict / 1.25)
            sum_pi = sum( [pi_predict[get_idx_output(G, move)] for move in moves] )
            if sum_pi == 0
                pi_predict = [1.0f0 / length(moves) for x in pi_predict]
                sum_pi = 1.0f0
            end
            for (i, move) in enumerate(moves)
                idx_output = get_idx_output(G, move)
                push!( cur.children, AZmodule.Node(move, G.playerToPlay, pi_predict[idx_output] / sum_pi, cur))
                cur.children[end].pi_dirichlet = dirichlet[i]
            end

        else
            Flux.softmax!(pi_predict)
            sum_pi = sum( [pi_predict[get_idx_output(G, move)] for move in moves] )
            if sum_pi == 0  # happens if all moves considered by NN are masked out because unplayable
                pi_predict = [1.0f0 / length(moves) for x in pi_predict]
                sum_pi = 1.0f0
            end
            for (i, move) in enumerate(moves)
                idx_output = get_idx_output(G, move)
                push!( cur.children, AZmodule.Node(move, G.playerToPlay,  pi_predict[idx_output] / sum_pi , cur))
            end
        end
    end

    while !isnothing(cur)
        cur.N += 1
        cur.w += winExpectancies[cur.justPlayedPlayer == 0 ? 1 : cur.justPlayedPlayer]          # If node is random, score as if it was to player 1
        cur.points += pointsExpectancies[cur.justPlayedPlayer == 0 ? 1 : cur.justPlayedPlayer]  # Same

        #=if AZmodule.player_of_tree(cur) == 1 && !isnothing(cur.parent)
            L = 0.995
            cur.pi_value = L*cur.pi_value + (1.0-L) / length(cur.parent.children)
        end=#

        cur = cur.parent
    end
end


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
end

function get_move_temperature(AZ::AZAlgorithm, G::Game.AbstractGame, temperature::Float32; rng=Random.default_rng(), forced_playouts=false)
    pi_MCTS::Vector{Float32} = zeros(NUMBER_ACTIONS)
    v_MCTS::Float32 = -1.0

    if AZ.ROOT.is_terminal == NOT_TERMINAL  # take a child with probability proportional to N^(1/temperature)
        v_MCTS = (G.playerToPlay == AZ.ROOT.justPlayedPlayer ? AZ.ROOT.w/AZ.ROOT.N : 1-AZ.ROOT.w/AZ.ROOT.N) #We want value for G.playerToPlay
        for child in AZ.ROOT.children if child.is_terminal != TERMINAL_LOSS
            idx_output = get_idx_output(G, child.justPlayedMove)
            pi_MCTS[idx_output] = forced_playouts ? max(0, child.N - sqrt(kFP * child.pi_value * child.parent.N)) : child.N
        end end


        if sum(pi_MCTS) == 0
            # happens if all children are 0 or loss
            # take all nodes even loses
            for child in AZ.ROOT.children
                idx_output = get_idx_output(G, child.justPlayedMove)
                pi_MCTS[idx_output] = child.N
            end
        end

        if sum(pi_MCTS) == 0
            print("root n : ", AZ.ROOT.N)
            print(AZ, G)
            print(G)
            error("Impossible davoir que des 0")
        end

        pi_MCTS /= sum(pi_MCTS)

        if temperature <= 0.05f0
            idx_move = argmax(pi_MCTS)
        elseif temperature == 1.0f0
            idx_move = rand(rng, Categorical(pi_MCTS))
        else
            P = pi_MCTS .^ (1.0f0/temperature)
            P /= sum(P)
            idx_move = rand(rng, Categorical(P))
        end

    elseif AZ.ROOT.is_terminal == (G.playerToPlay == AZ.ROOT.justPlayedPlayer ? TERMINAL_LOSS : TERMINAL_WIN)     # all moves are loss : take a child at random
        v_MCTS = 0.0
        for child in AZ.ROOT.children
            idx_output = get_idx_output(G, child.justPlayedMove)
            pi_MCTS[ idx_output ] = 1 / length(AZ.ROOT.children)
        end
        idx_move = rand(rng, Categorical(pi_MCTS))

    elseif AZ.ROOT.is_terminal == TERMINAL_DRAW    # all moves are draws : take a draw child at random
        v_MCTS = 0.5
        sum_pi = 0
        for child in AZ.ROOT.children if child.is_terminal == TERMINAL_DRAW
            idx_output = get_idx_output(G, child.justPlayedMove)
            pi_MCTS[ idx_output ] = 1
            sum_pi += 1
        end end
        pi_MCTS /= sum_pi
        idx_move = rand(rng, Categorical(pi_MCTS))

    else            # some moves are wins : take one at random
        v_MCTS = 1.0
        sum_pi = 0
        lowest_depth = INF
        for child in AZ.ROOT.children if child.is_terminal == TERMINAL_WIN
            if AZmodule.depth(child) == lowest_depth
                idx_output = get_idx_output(G, child.justPlayedMove)
                pi_MCTS[ idx_output ] = 1
                sum_pi += 1
            elseif AZmodule.depth(child) < lowest_depth
                lowest_depth = AZmodule.depth(child)
                pi_MCTS *= 0
                idx_output = get_idx_output(G, child.justPlayedMove)
                pi_MCTS[ idx_output ] = 1
                sum_pi = 1
            end
        end end
        pi_MCTS /= sum_pi
        idx_move = rand(rng, Categorical(pi_MCTS))
    end

    move = get_move_from_idx_move(G, idx_move)

    return move, pi_MCTS, v_MCTS
end=#

end
