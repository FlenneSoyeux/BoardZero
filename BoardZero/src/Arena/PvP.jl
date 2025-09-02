"""
    PvP.jl

    extension of Arena.jl
    code for every PvP possibilities,
    P in {human, MCTS, AZ} gives 6 functions to implement
"""



#=
function _save_game(first_player::Int, history_move::Vector{AbstractMove})
    open(DIRECTORY*"last_game.txt", "w") do f
        write(f, string(first_player)*"\n")
        _G = new_game()
        for m in history_move
            write(f, Game.move_to_string(_G, m)*"\n")
            play!(_G, m)
        end
        close(f)
    end
end

function _load_game()
    G = new_game()
    history_move::Vector{AbstractMove} = []

    if isfile(DIRECTORY*"last_game.txt")
        println("Load previous game? (type 'y') ")
        if readline() in ["y", "Y"]
            file = open(DIRECTORY*"last_game.txt", "r")
            first_player = parse(Int64, readline(file))
            println("first_player est ", first_player)
            for ln in eachline(file)    #structure de chaque ligne est : "move: string(move)"
                println("line : ", ln)
                push!(history_move, Game.string_to_move(G, ln))
                play!(G, history_move[end])
            end
            close(file)

        else
            first_player = rand(1:2)
        end

    else
        first_player = rand(1:2)
    end

    return first_player, G, history_move
end=#


function human_vs_AZ(maxIteration::Int = 0, timeLimit::Float64 = 1.0)
#     human_player, G, history_move = _load_game()
    human_player = rand(1:2)
    G = new_game()


    nn = are_all_nn_set() ? initialize_model(:ELO) : initialize_model(:last)
    az = AZAlgorithm(G, nn, maxIteration, timeLimit)

    while !is_finished(G)
        if G.playerToPlay == 0
            move = get_move_random_stuffs(G)
            play!(G, move)
            keep_subtree!(az, move)
        end

        print(G)
        if G.playerToPlay == human_player
            move = get_human_move(G)
        else
            move = get_move(az, G, 0.0f0)
            print(az, G)
        end

        println("move : ", get_printable_move(G, move))
#         push!(history_move, move)
        #_save_game(human_player, history_move)
        play!(G, move)
        keep_subtree!(az, move)
        println("On a gardé : ", az.ROOT.N, " nodes")

        GC.gc(true)
    end
    print(G)
end


function MCTS_vs_AZ(maxIteration_mcts::Int = 0, timeLimit_mcts::Float64 = 1.0, maxIteration_az::Int = maxIteration_mcts, timeLimit_az::Float64 = timeLimit_mcts)
#     mcts_player, G = _load_game()
    mcts_player = rand(1:NUM_PLAYERS)
    G = new_game()

    nn = initialize_model(:last)
    az = AZAlgorithm(G, nn, maxIteration_az, timeLimit_az)
    mcts = MCTSAlgorithm(G, maxIteration_mcts, timeLimit_mcts)
    history_move::Vector{AbstractMove} = []

    while !is_finished(G)
        if G.playerToPlay == 0
            move = get_move_random_stuffs(G)
            play!(G, move)
            keep_subtree!(az, move)
            keep_subtree!(mcts, move)
        end
        print(G)
        if G.playerToPlay == mcts_player
            move = get_move(mcts, G)
            print(mcts, G)
        else
            move = get_move(az, G)
            print(az, G)
        end
        println("move : ", get_printable_move(G, move))
#         push!(history_move, move)
#         _save_game(mcts_player, history_move)
        play!(G, move)
        keep_subtree!(az, move)
        keep_subtree!(mcts, move)
    end
    print(G)
end


function human_vs_MCTS(maxIteration::Int = 0, timeLimit::Float64 = 1.0)
    G = new_game()
    mcts = MCTSAlgorithm(G, maxIteration, timeLimit)
    human_player = rand(1:NUM_PLAYERS)

    while !is_finished(G)
        print(G)
        if G.playerToPlay == human_player
            move = get_human_move(G)
        else
            move = get_move(mcts, G)
            print(mcts, G)
        end
        play!(G, move)
        keep_subtree!(mcts, move)
        println("move : ", get_printable_move(G, move))
    end
    print(G)
end

function moves_helper(maxIteration::Int = 0, timeLimit::Float64 = 1.0; newGame=false, useMCTS=false)
    G = new_game()
    if useMCTS
        agent = MCTSAlgorithm(G, maxIteration, timeLimit)
        nodes = MCTSmodule.Node[]
    else
        nn = are_all_nn_set() ? initialize_model(:ELO) : initialize_model(:last)
        agent = AZAlgorithm(G, nn, maxIteration, timeLimit)
        nodes = AZmodule.Node[]
    end
    manual_input(G; newGame=newGame)

    history_games = AbstractGame[]
    

    while !is_finished(G)
        if G.playerToPlay == 0
            print(G)
            push!(history_games, deepcopy(G))
            get_human_random_stuffs(G)
            print(G)
            keep_subtree!(agent, get_null_move(G))
        end

        doCompute = (agent.ROOT.N == 0)
        doPrint = true
        doStop = false
        while !doStop
            if doCompute
                println("Computing...")
                get_move(agent, G)
            end
            if doPrint
                print(agent, G)
            end
            print("'c' to continue searching; 's' to stop searching; 'e' to expand printing; 'ee' to double expand printing")
            println(length(history_games)>0 ? "'-1' to get back" : "")
            str = readline()
            if str == "s"
                doStop = true
                
            elseif length(str) >= 1 && str[1] == 'e'
                print(agent, G; depth_child=length(str))
                doStop = false
                doCompute = false
                doPrint = false

            elseif str == "c"
                agent.maxIteration += maxIteration
                doStop = false
                doCompute = true
                doPrint = true

            elseif length(history_games)>0 && str == "-1"
                G = history_games[end]
                pop!(history_games)
                if G.playerToPlay == 0
                    G = history_games[end]
                    pop!(history_games)
                end

                agent.ROOT.parent = nodes[end]
                agent.ROOT = nodes[end]
                agent.ROOT.N = 1 + sum([c.N for c in agent.ROOT.children])
                agent.ROOT.w = (useMCTS ? 0 : agent.ROOT.v_initial[1]) + sum([c.w for c in agent.ROOT.children])
                if (agent.ROOT.justPlayedPlayer == 0 && agent.ROOT.children[1].justPlayedPlayer == 2) ||
                        (agent.ROOT.justPlayedPlayer != 0 && agent.ROOT.justPlayedPlayer != agent.ROOT.children[1].justPlayedPlayer)
                    agent.ROOT.w = agent.ROOT.N - agent.ROOT.w
                end

                print(agent, G)
                pop!(nodes)                
                print(G)

                doStop = false
                doCompute = false
                doPrint = false

            else
                doStop = false
                doCompute = false
                doPrint = false

            end
        end
        print(G)
        move = get_human_move(G)
        push!(history_games, deepcopy(G))
        play!(G, move)
        print(G)
        push!(nodes, agent.ROOT)
        keep_subtree!(agent, move)
    end

    print(G)
end


function MCTS_vs_MCTS(maxIteration::Int = 0, timeLimit::Float64 = 1.0)
    G = new_game()
    mcts = MCTSAlgorithm(G, maxIteration, timeLimit)

    while !is_finished(G)

        if G.playerToPlay == 0
            move = get_move_random_stuffs(G)
            play!(G, move)
            keep_subtree!(mcts, move)
            continue
        end
        
        print(G)
        move = get_move(mcts, G)
        print(mcts, G)
        println("move : ", get_printable_move(G, move))
        play!(G, move)
        keep_subtree!(mcts, move)
        GC.gc(true)
    end
    print(G)
end

function AZ_vs_AZ(maxIteration::Int, timeLimit::Float64, pathToNNA, pathToNNB)
    G = new_game()
    nna = initialize_model(pathToNNA)
    nnb = initialize_model(pathToNNB)
    aza = AZAlgorithm(G, nna, maxIteration, timeLimit)
    azb = AZAlgorithm(G, nnb, maxIteration, timeLimit)
    playera = 1

    while !is_finished(G)
        print(G)
        if G.playerToPlay == 0
            move = get_move_random_stuffs(G)

        elseif G.playerToPlay == playera
            move = get_move(aza, G)
            print(aza, G)

        else
            move = get_move(azb, G)
            print(azb, G)
        end
        println("move : ", get_printable_move(G, move))
        play!(G, move)
        keep_subtree!(aza, move)
        keep_subtree!(azb, move)
        GC.gc(true)
    end
    print(G)
end

function AZ_vs_AZ(maxIteration::Int = 0, timeLimit::Float64 = 1.0)
    G = new_game()
    nn = initialize_model(:last)
    az = AZAlgorithm(G, nn, maxIteration, timeLimit)

    while !is_finished(G)
        print(G)

        if G.playerToPlay == 0
            print(az, G)
            move = get_move_random_stuffs(G)
            play!(G, move)
            keep_subtree!(az, move)

        else
            move = get_move(az, G)
            print(az, G; top=10)
            println("move : ", get_printable_move(G, move))
            play!(G, move)
            keep_subtree!(az, move)

            if length(az.ROOT.children) > 0 && az.ROOT.children[1].justPlayedPlayer == 0
                println("ON Y EST")
                print(az, G)
            end
            
        end
        
        GC.gc(true)
    end
    print(G)
end

function human_vs_human()
    G = new_game()
    while !is_finished(G)
        print(G)
        move = get_human_move(G)
        play!(G, move)
        println("move : ", get_printable_move(G, move))
    end
    print(G)
end
