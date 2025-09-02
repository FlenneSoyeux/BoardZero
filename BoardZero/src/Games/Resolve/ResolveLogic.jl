
function check_win!(G::Resolve)
    function check_win_color(board, NorthSouthSides=true)    # player1 must join northsouth sides ; player2 east-west 
        visited = .! board  # trick : if !board[cell], mark cell as already visited. Already visited cells are not visited again.  

        function dfs(cell, visited, NorthSouthSides)
            if NorthSouthSides
                if cell%5 == 0
                    return true
                end
            elseif cell >= 21
                return true
            end
            visited[cell] = true
            for dir in 1:4
                if DIR_POSSIBLE[cell, dir] && !visited[cell+DIR[dir]]
                    if dfs(cell+DIR[dir], visited, NorthSouthSides)
                        return true
                    end
                end
            end
            return false
        end

        for i in 1:5
            cell = NorthSouthSides ? 5*i - 4 : i    # check cells 1 6 11 16 21 or 1 2 3 4 5
            if !visited[cell]
                if dfs(cell, visited, NorthSouthSides)
                    return true
                end
            end
        end
        return false
    end
        
    if check_win_color(G.board[:, 1], true)
        G.winner = 1
        G.isFinished = true
    elseif check_win_color(G.board[:, 2], false)
        G.winner = 2
        G.isFinished = true
    end
end

function Game.play!(G::Resolve, move::ResolveMove)
    if move.kind == 0
        G.lastMove = move.cell
        G.lastLastCell = 0
        G.board[move.cell, G.playerToPlay] = true
        G.remainingCells -= 1

    else
        swapCell = move.cell + DIR[move.kind]
        @assert 1 <= swapCell <= 25 string(move.cell)*" "*string(swapCell)*" "*string(move.kind)
        G.board[move.cell, G.playerToPlay] = false
        G.board[swapCell, G.playerToPlay] = true
        G.board[swapCell, 3-G.playerToPlay] = false
        G.board[move.cell, 3-G.playerToPlay] = true
        
        G.lastMove = swapCell
        G.lastLastCell = move.cell
    end

    check_win!(G)

    if G.isFinished
        # nothing
    else    #check for resolve before changing player
        resolve = false
        for dir in 1:4
            if !DIR_POSSIBLE[G.lastMove, dir] || !DIR_POSSIBLE[G.lastMove, dir%4+1]
                continue
            end
            da, db = DIR[dir], DIR[dir%4 + 1]
            if G.board[G.lastMove, G.playerToPlay] && G.board[G.lastMove+da, 3-G.playerToPlay] && G.board[G.lastMove+da+db, G.playerToPlay] && G.board[G.lastMove+db, 3-G.playerToPlay]
                #if G.lastMove+da == G.lastLastCell || G.lastMove+db == G.lastLastCell   # can't do swaps A->B->A
                #    continue
                #end
                resolve = true
                break
            end
        end

        if resolve && G.turnSamePlayer == 1
            G.isFinished = true
            G.winner = 0
        elseif resolve
            G.turnSamePlayer -= 1
        else
            G.playerToPlay = 3-G.playerToPlay
            G.turnSamePlayer = MAX_TURN_SAME_PLAYER
            G.lastLastCell = 0
        end
    end
end

function Game.all_moves(G::Resolve) :: Vector{ResolveMove}
    moves = ResolveMove[]
    if G.lastMove != 0 && G.board[G.lastMove, G.playerToPlay] # current player has played the last move : some resolve has been detected !
        for dir in 1:4
            if !DIR_POSSIBLE[G.lastMove, dir] || !DIR_POSSIBLE[G.lastMove, dir%4+1]
                continue
            end
            da, db = DIR[dir], DIR[dir%4 + 1]
            if G.board[G.lastMove, G.playerToPlay] && G.board[G.lastMove+da, 3-G.playerToPlay] && G.board[G.lastMove+da+db, G.playerToPlay] && G.board[G.lastMove+db, 3-G.playerToPlay]
                #if G.lastMove+da == G.lastLastCell || G.lastMove+db == G.lastLastCell # can't do swaps A->B->A
                #    continue
                #end
                if G.lastMove+da != G.lastLastCell && !(ResolveMove(G.lastMove, dir) in moves)
                    push!(moves, ResolveMove(G.lastMove, dir))
                end
                if G.lastMove+db != G.lastLastCell && !(ResolveMove(G.lastMove, dir%4 + 1) in moves)
                    push!(moves, ResolveMove(G.lastMove, dir%4 + 1))
                end
            end
        end
        
    elseif G.remainingCells == 0 # Look for resolve to do
        for x in 2:SIZE, y in 2:SIZE
            # check on squares (x,y) (x-1,y) (x-1,y-1) (x,y-1) (up(1) and left(2) from xy)
            cella = xy2cell(x, y)
            cellb = xy2cell(x-1, y)
            cellc = xy2cell(x-1, y-1)
            celld = xy2cell(x, y-1)
            if (G.board[cella, 1] && G.board[cellb, 2] && G.board[cellc, 1] && G.board[celld, 2]) ||
                 (G.board[cella, 2] && G.board[cellb, 1] && G.board[cellc, 2] && G.board[celld, 1])
                ResolveMove(cella, 1) in moves || push!(moves, ResolveMove(cella, 1))   # up from x,y
                ResolveMove(cella, 2) in moves || push!(moves, ResolveMove(cella, 2))   # left from x,y
                ResolveMove(celld, 1) in moves || push!(moves, ResolveMove(celld, 1))   # up from x,y-1
                ResolveMove(cellb, 2) in moves || push!(moves, ResolveMove(cellb, 2))   # left from x-1,y
            end
        end

    else
        for cell in 1:25
            if !G.board[cell, 1] && !G.board[cell, 2]
                push!(moves, ResolveMove(cell, 0))
            end
        end
    end

    return moves
end


function Game.get_human_move(G::Resolve)
    moves = all_moves(G)
    if moves[1].kind != 0   # Resolve !
        for (i, m) in enumerate(moves)
            println(i, ":", get_printable_move(G, m))
        end
        i = parse(Int64, readline())
        return moves[i]
    else  # place !
        cell = -1
        while cell < 1 || cell > 25 || G.board[cell, 1] || G.board[cell, 2]
            try
                x, y = split(readline(), ' ')
                x = parse(Int, x)
                y = parse(Int, y)
                cell = xy2cell(x, y)
            catch ex
                if isa(ex, InterruptException)
                    rethrow(ex)
                end
            end
        end
        return ResolveMove(cell, 0)
    end
end


function Game.play_random!(G::Resolve, rng=default_rng())
    while !is_finished(G)
        moves = all_moves(G)
        if length(moves) == 0
            print(G)
            error("null")
        end
        move = rand(rng, moves)
        play!(G, move)
    end
end