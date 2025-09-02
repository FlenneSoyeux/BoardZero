
function can_move(G::Santorini) :: Bool #can the player move the 2 workers ?
    if G.gods[G.playerToPlay] == PROMETHEUS && G.godFlag[G.playerToPlay] != 0   # PROMETHEUS : a worker built before move. It must now move equal or down
        idWorker = G.godFlag[G.playerToPlay]
        here = G.workers[G.playerToPlay, idWorker]
        for next in NEIGHBORS[here]
            if G.board[next] <= G.board[here] &&
                    G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                return true
            end
        end
        return false
    end


    for id in 1:2
        here = G.workers[G.playerToPlay, id]
        for next in NEIGHBORS[here]
            if G.gods[3-G.playerToPlay] == ATHENA && G.godFlag[3-G.playerToPlay] == 1   # ATHENA : opponent cant move up
                if G.board[next] <= G.board[here] &&       
                        G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                    return true
                end

            elseif G.board[next] - G.board[here] <= 1 && G.board[next] <= 3 &&  # normal move
                    G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                return true

            elseif G.gods[G.playerToPlay] == APOLLO && G.board[next] - G.board[here] <= 1 && G.board[next] < 4 &&       # APOLLO power : can move to enemy (to swap)
                    G.workers[G.playerToPlay,1] != next && G.workers[G.playerToPlay,2] != next
                return true

            elseif G.gods[G.playerToPlay] == MINOTAUR && G.board[next] - G.board[here] <= 1 && G.board[next] < 4 &&       # APOLLO power : can move to enemy (to swap)
                    G.workers[G.playerToPlay,1] != next && G.workers[G.playerToPlay,2] != next && bumpable_minotaur(here, next, G)
                return true
            end
        end
    end

    return false
end

function can_build(G::Santorini) :: Bool #can the player move the 2 workers ?
    if G.gods[G.playerToPlay] == DEMETER && G.godFlag[G.playerToPlay] != 0
        return true # can always pass
    end
    here = G.workers[G.playerToPlay, G.buildWorker]
    for next in NEIGHBORS[here]
        if G.board[next] <= 3 &&
            G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
            return true
        end
    end

    return false
end

function Game.play!(G::Santorini, move::SantoriniMove)
    if G.buildWorker == INITIALIZATION
        G.workers[G.playerToPlay, move.idWorker] = move.cell
        if move.idWorker == 2
            if sum(G.workers .!= 0) == 4
                G.buildWorker = 0   # change state : now move
                G.playerToPlay = 3-G.playerToPlay
            else
                G.playerToPlay = 3-G.playerToPlay
            end
        end

    elseif move.kind == ACTION_MOVE  
        if G.gods[G.playerToPlay] == APOLLO && (G.workers[3-G.playerToPlay, 1] == move.cell || G.workers[3-G.playerToPlay, 2] == move.cell) #APOLLO
            # make a swap
            if G.workers[3-G.playerToPlay, 1] == move.cell
                G.workers[3-G.playerToPlay, 1] = G.workers[G.playerToPlay, move.idWorker]
                G.workers[G.playerToPlay, move.idWorker] = move.cell
            elseif G.workers[3-G.playerToPlay, 2] == move.cell
                G.workers[3-G.playerToPlay, 2] = G.workers[G.playerToPlay, move.idWorker]
                G.workers[G.playerToPlay, move.idWorker] = move.cell
            end
        elseif G.gods[G.playerToPlay] == MINOTAUR && (G.workers[3-G.playerToPlay, 1] == move.cell || G.workers[3-G.playerToPlay, 2] == move.cell) #MINOTAUR
            # bump
            nextnext = 2*move.cell - G.workers[G.playerToPlay, move.idWorker]
            if G.workers[3-G.playerToPlay, 1] == move.cell
                G.workers[3-G.playerToPlay, 1] = nextnext
                G.workers[G.playerToPlay, move.idWorker] = move.cell
            elseif G.workers[3-G.playerToPlay, 2] == move.cell
                G.workers[3-G.playerToPlay, 2] = nextnext
                G.workers[G.playerToPlay, move.idWorker] = move.cell
            end
        else    # SIMPLE move
            if G.gods[G.playerToPlay] == ATHENA 
                G.godFlag[G.playerToPlay] = (G.board[G.workers[G.playerToPlay, move.idWorker]] < G.board[move.cell])  # ATHENA : opponent cant move up
            elseif G.gods[G.playerToPlay] == HERMES
                if G.godFlag[G.playerToPlay] == 1
                    G.godFlag[G.playerToPlay] = 0
                else
                    G.godFlag[G.playerToPlay] = (G.board[G.workers[G.playerToPlay, move.idWorker]] == G.board[move.cell])
                end
            elseif G.gods[G.playerToPlay] == PAN && G.board[move.cell] == 0 && G.board[G.workers[G.playerToPlay, move.idWorker]] == 2
                G.isFinished = true
                G.winner = G.playerToPlay
            end
                
            G.workers[G.playerToPlay, move.idWorker] = move.cell
        end

        if G.gods[G.playerToPlay] == PROMETHEUS
            G.godFlag[G.playerToPlay] = 0
        end

        G.buildWorker = move.idWorker  # change state : now idWorker must build
        if G.board[ move.cell ] == 3
            G.isFinished = true
            G.winner = G.playerToPlay
        end

    elseif move.kind == ACTION_BUILD
        G.board[ move.cell ] += 1
        if G.gods[G.playerToPlay] == DEMETER
            if G.godFlag[G.playerToPlay] == 0   # another build
                G.godFlag[G.playerToPlay] = move.cell
            else    # change state and player
                G.godFlag[G.playerToPlay] = 0
                G.buildWorker = 0
                G.playerToPlay = 3-G.playerToPlay
            end
        elseif G.gods[G.playerToPlay] == PROMETHEUS && G.buildWorker == 0
            G.godFlag[G.playerToPlay] = move.idWorker
        else    # Normal build
            G.buildWorker = 0   # change state
            G.playerToPlay = 3-G.playerToPlay
        end

        if G.gods[3-G.playerToPlay] == HERMES
            G.godFlag[3-G.playerToPlay] = 0   # reset flag if player did not use ability
        end

        
    elseif move.kind == ACTION_SPECIAL
        # for ATLAS or DEMETER
        if G.gods[G.playerToPlay] == ATLAS  # build dome
            G.board[ move.cell ] = 4
            G.buildWorker = 0   # change state
            G.playerToPlay = 3-G.playerToPlay

        elseif G.gods[G.playerToPlay] == DEMETER    # pass action
            G.buildWorker = 0   # change state
            G.godFlag[G.playerToPlay] = 0
            G.playerToPlay = 3-G.playerToPlay
        elseif G.gods[G.playerToPlay] == HEPHAESTUS # build twice
            G.board[ move.cell ] += 2
            G.buildWorker = 0
            G.playerToPlay = 3-G.playerToPlay
        else
            error("not good god", G.gods[G.playerToPlay])
        end

    end


    # Other win conditions :
    if G.buildWorker == 0 && !can_move(G)
        G.isFinished = true
        G.winner = 3-G.playerToPlay
    end
    if G.buildWorker >= 1 && !can_build(G)
        G.isFinished = true
        G.winner = 3-G.playerToPlay
    end
end

using DataStructures

function dfs_hermes(visited::BitVector, cell::Int, all_moves::Vector{SantoriniMove}, idWorker::Int, G::Santorini)    # add to moves all the reachable cells of same level
    visited[cell] = true
    push!(all_moves, SantoriniMove(idWorker, cell, ACTION_MOVE))
    for next in NEIGHBORS[cell]
        if !visited[next] && G.board[next] == G.board[cell]
            dfs_hermes(visited, next, all_moves, idWorker, G)
        end
    end
end

function bumpable_minotaur(here::Int, next::Int, G::Santorini)
    @assert G.workers[3-G.playerToPlay, 1] == next || G.workers[3-G.playerToPlay, 2] == next
    nextnext = 2*next - here
    if !(nextnext in NEIGHBORS[next])   # against wall
        return false
    end
    return G.board[nextnext] < 4 && G.workers[1, 1] != nextnext && G.workers[1, 2] != nextnext && G.workers[2, 1] != nextnext && G.workers[2, 2] != nextnext
end

function Game.all_moves(G::Santorini) :: Vector{SantoriniMove}
    all_moves = Vector{SantoriniMove}()
    ### INITIALIZATION : Players place the workers
    if G.buildWorker == INITIALIZATION
        idWorker = (G.workers[G.playerToPlay, 1] == 0) ? 1 : 2
        for cell in 1:25
            if G.workers[1, 1] != cell && G.workers[1, 2] != cell && G.workers[2, 1] != cell && G.workers[2, 2] != cell
                push!(all_moves, SantoriniMove( idWorker, cell, ACTION_MOVE ))
            end
        end
    end

    ### MOVE ACTIONS
    if G.buildWorker == 0 || (G.gods[G.playerToPlay] == HERMES && G.godFlag[G.playerToPlay] == 1)
        if G.gods[G.playerToPlay] == ARTEMIS    # ARTEMIS : moves once or twice
            for idWorker in 1:2
                visited = (G.board .== 4)   # dont go on already visited nodes : why not add 4 levels and other forbidden cells ?
                for p in 1:2, i in 1:2
                    visited[G.workers[p, i]] = true
                end
                q = Queue{Tuple{Int, Int}}()
                enqueue!(q, (G.workers[G.playerToPlay, idWorker], 2))
                while !isempty(q)
                    cell, nmove = dequeue!(q)
                    for next in NEIGHBORS[cell]
                        if !visited[next] && G.board[next] - G.board[cell] <= 1
                            visited[next] = true
                            nmove == 1 || enqueue!(q, (next, nmove-1))
                            push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                        end
                    end
                end
            end
            return all_moves
        elseif G.gods[G.playerToPlay] == HERMES
            if G.godFlag[G.playerToPlay] == 0   # 1 moves on equal levels or different levels or same square, 2 moves on equal or different or same square
                for idWorker in 1:2
                    here = G.workers[G.playerToPlay, idWorker]
                    visited = (G.board .== 4)
                    for p in 1:2, i in 1:2
                        visited[ G.workers[p, i] ] = true
                    end
                    dfs_hermes(visited, here, all_moves, idWorker, G)   # adds all the moves of same level
                    for next in NEIGHBORS[here] # also add all the regular moves
                        if !visited[next] && G.board[next] - G.board[here] <= 1
                            push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                        end
                    end
                end
            else    # 1 (actually:G.buildWorker) has moved on equal : 2 must move on equal or 1 must build immediately
                if G.buildWorker <= 0
                    print(G)
                    error("mais non")
                end
                idWorker = 3-G.buildWorker
                here = G.workers[G.playerToPlay, idWorker]
                visited = (G.board .== 4)
                for p in 1:2, i in 1:2
                    visited[ G.workers[p, i] ] = true
                end
                dfs_hermes(visited, here, all_moves, idWorker, G)   # adds all the moves of same level
            end
        elseif G.gods[G.playerToPlay] == PROMETHEUS && G.godFlag[G.playerToPlay] != 0
            # idWorker has already built. It must now move on equal or lower levels
            idWorker = G.godFlag[G.playerToPlay]
            here = G.workers[G.playerToPlay, idWorker]
            for next in NEIGHBORS[here]
                if G.board[next] <= G.board[here] &&
                        G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                    push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                end
            end

        else
            for idWorker in 1:2
                here = G.workers[G.playerToPlay, idWorker]
                for next in NEIGHBORS[here]
                    if G.gods[3-G.playerToPlay] == ATHENA && G.godFlag[3-G.playerToPlay] == 1    # ATHENA : opponent cant move up
                        if G.board[next] <= G.board[here] && G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                            push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                        end
                    elseif G.board[next] - G.board[here] <= 1 && G.board[next] < 4 &&   # normal move
                            G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                        push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                    elseif G.gods[G.playerToPlay] == APOLLO && G.board[next] - G.board[here] <= 1 && G.board[next] < 4 &&       # APOLLO power : can move to enemy (to swap)
                            G.workers[G.playerToPlay,1] != next && G.workers[G.playerToPlay,2] != next
                        # for sure there is opponent on next
                        push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                    elseif G.gods[G.playerToPlay] == MINOTAUR && G.board[next] - G.board[here] <= 1 && G.board[next] < 4 &&       # MINOTAUR power : can bump
                            G.workers[G.playerToPlay,1] != next && G.workers[G.playerToPlay,2] != next
                        # for sure there is opponent on next
                        if bumpable_minotaur(here, next, G)
                            push!(all_moves, SantoriniMove(idWorker, next, ACTION_MOVE))
                        end
                    end
                end
            end
        end
    end

    ### BUILD ACTIONS
    if G.gods[G.playerToPlay] == PROMETHEUS && G.buildWorker == 0 && G.godFlag[G.playerToPlay] == 0
        for idWorker in 1:2
            here = G.workers[G.playerToPlay, idWorker]
            for next in NEIGHBORS[here]
                if G.board[next] < 4 && G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                    push!(all_moves, SantoriniMove(idWorker, next, ACTION_BUILD))
                end
            end
        end

    elseif 1 <= G.buildWorker <= 2
        here = G.workers[G.playerToPlay, G.buildWorker]
        for next in NEIGHBORS[here]
            if G.gods[G.playerToPlay] == DEMETER && G.godFlag[G.playerToPlay] == next
                push!(all_moves, SantoriniMove(G.buildWorker, here, ACTION_SPECIAL))    # PASS action
                continue
            end
            if G.board[next] < 4 && G.workers[1,1] != next && G.workers[1,2] != next && G.workers[2,1] != next && G.workers[2,2] != next
                push!(all_moves, SantoriniMove(G.buildWorker, next, ACTION_BUILD))
                if G.gods[G.playerToPlay] == ATLAS && G.board[next] < 3  # build dome on 0 1 2 levels
                    push!(all_moves, SantoriniMove(G.buildWorker, next, ACTION_SPECIAL))
                elseif G.gods[G.playerToPlay] == HEPHAESTUS && G.board[next] < 2    # build double on 0 1 levels  
                    push!(all_moves, SantoriniMove(G.buildWorker, next, ACTION_SPECIAL))
                end
            end
        end
    end

    if length(all_moves) == 0
        print(G)
        error("noooo222222")
    end
    return all_moves
end


function Game.play_random!(G::Santorini, rng = Random.default_rng())
    while !is_finished(G)
        move = rand(rng, all_moves(G))
        play!(G, move)
    end
end


function Game.get_human_move(G::Santorini) :: SantoriniMove
    moves = all_moves(G)
    for (i, m) in enumerate(moves)
        println(i, "\t", get_printable_move(G, m))
    end

    choice = 0
    while choice < 1 || choice > length(moves)
        println("Choix ? ")
        choice = parse(Int, readline())
    end
    return moves[choice]
end

