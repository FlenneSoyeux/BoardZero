#Abstract Games

module Game

export AbstractGame, AbstractRandomGame, AbstractPointsGame, AbstractMove, new_game
export play!, all_moves, is_finished, get_printable_move, get_winner, play_random!, get_null_move, random_move, stopTreeExpansion, get_utility_score, points_to_utility, get_move_random_stuffs, get_human_random_stuffs
export get_human_move, manual_input       # for HumanAlgorithm
export get_idx_output, get_move_from_idx_move, load_from , get_input_for_nn, print_input_nn      # for AZ
export TYPE_GAME, BLANK_GAME

abstract type AbstractMove end
abstract type AbstractGame end
abstract type AbstractRandomGame <: AbstractGame end
abstract type AbstractPointsGame <: AbstractGame end

# Usually put in a file named ...Game.jl 

function is_finished(G::AbstractGame) :: Bool
    error("is_finished in abstractgame")
end

function get_winner(G::AbstractGame)
    # 1 or 2 ; 0 for draw ; WARNING : asserted that game is finished
    error("get_winner not for abstractgame")
end

function Base.print(G::AbstractGame) :: Nothing
    error("print not for abstractgame")
end



function get_printable_move(G::AbstractGame, move::AbstractMove) :: String
    error("get_printable_move not for abstractgame")
end
# for saving in file :
function string_to_move(G::AbstractGame, s::String) :: AbstractMove
    error("string_to_move not for abstractgame")
end
function move_to_string(G::AbstractGame, s::AbstractMove) :: String
    error("move_to_string not for abstractmove")
end

function get_null_move(G::AbstractGame) :: AbstractMove
    error("get_null_move not for abstractgame")
end

function get_utility_score(G::AbstractPointsGame) :: Float32
    error("get_utility_score not for abstractpointsgame")
end

function points_to_utility(G::AbstractGame, points) :: Float32
    return 0
    #error("points_to_utility ot for avbstractgame")
end

function number_played_moves(G::AbstractGame) :: Int
    error("number_played_moves not for abstractgame")
end

function get_delta_points(G::AbstractGame)  # for player 1
    return 0.0f0
end







# Usually put in a file named ...Logic.jl
function play!(G::AbstractGame, move::AbstractMove) :: Nothing
    error("cant play abstractgame")
end

function all_moves(G::AbstractGame) :: Vector{AbstractMove}
    error("all_moves not for abstractgame")
end

function get_human_move(G::AbstractGame) :: AbstractMove
    error("get_human_move not for abstractgame")
end

function get_human_random_stuffs(G::AbstractGame)
    error("get_human_random_stuffs not for abstractGame")
end

function play_random!(G::AbstractGame, rng) :: Nothing
    error("play_random not for abstractgame")
end

function random_move(G::AbstractGame, rng) :: AbtractMove
    error("Random_move not for abstractgame")
end

function get_move_random_stuffs(G::AbstractGame, rng)
    error("get_move_random_stuffs not for abstractgame")
end

function manual_input(G::AbstractGame)
    error("manual_input not for abstractgame")
end


# Usually put in a file named ...NN.jl. Also add in ...NN.jl all functions from NNet.jl
function load_from(Gfrom::AbstractGame, Gto::AbstractGame)
    error("cant load abstractgame")
end

# Sometimes we can't perform output_nn[move] directly. We need to find idx to do output_nn[idx] after
function get_idx_output(G::AbstractGame, move::AbstractMove) :: Int
    error("get_idx_output not for abstractgame")
end

# After output of nn, we chose move number idx. What move is it corresponding to ?
function get_move_from_idx_move(G::AbstractGame, idx::Int) :: AbstractMove
    error("get_move_from_idx_move not for abstractgame")
end

function get_input_for_nn(G::AbstractGame)
    error("get_input_for_nn not for abstractgame")
end
function get_input_for_nn!(G::AbstractGame, pi_MCTS::Vector{Float32}, rng)
    error("get_input_for_nn_2 not for abstractgame")
end

function print_input_nn(::AbstractGame, input, pi_MCTS; print_pos=true)
    error("print input not for abstractgame")
end




using Crayons, Random, DataStructures

include("../params.jl")

@static if GAME == "Gomoku6" || GAME == "Gomoku9" || GAME == "Gomoku13" || GAME == "Gomoku15"
    include("Gomoku.jl")
    export Gomoku
elseif GAME == "Pente"
    include("Pente.jl")
    export Pente
elseif GAME == "Santorini"
    include("Santorini/SantoriniGame.jl")
    include("Santorini/SantoriniLogic.jl")
    export Santorini, SantoriniMove, ACTION_BUILD, ACTION_MOVE, ACTION_SPECIAL, cell2xy, xy2cell
elseif GAME == "Connect4"
    include("Connect4.jl")
    export Connect4
elseif GAME == "Kalah"
    include("Kalah.jl")
    export Kalah
elseif GAME == "Boop"
    include("Boop/BoopGame.jl")
    include("Boop/BoopLogic.jl")
    export Boop, BoopMove, cells, CHAR
elseif GAME == "Jaipur"
    include("Jaipur.jl")
    export Jaipur
elseif GAME == "Azul"
    include("Azul/AzulGame.jl")
    include("Azul/AzulLogic.jl")
    export Azul, AzulMove, get_raw_points_round, BOARDCOLOR, CENTER, COLORPOS, ACTION_RANDOM, ACTION_TAKE, ACTION_PLACE
elseif GAME == "Azul2"
    include("Azul2.jl")
    export Azul, get_raw_points_round
elseif GAME == "Resolve"
    include("Resolve/ResolveGame.jl")
    include("Resolve/ResolveLogic.jl")
    export Resolve, ResolveMove, cell2xy, xy2cell
else
    error("Game not found : "*GAME)
end

function new_game() :: AbstractGame
    @static if GAME == "Gomoku6" || GAME == "Gomoku9" || GAME == "Gomoku13" || GAME == "Gomoku15"
        return Gomoku()
    elseif GAME == "Connect4"
        return Connect4()
    elseif GAME == "Pente"
        return Pente()
    elseif GAME == "Santorini"
        return Santorini()
    elseif GAME == "Kalah"
        return Kalah()
    elseif GAME == "Boop"
        return Boop()
    elseif GAME == "Jaipur"
        return Jaipur()
    elseif GAME == "Azul"
        return Azul()
    elseif GAME == "Resolve"
        return Resolve()
    else
        error("???")
    end
end

const TYPE_GAME = typeof(new_game())
const BLANK_GAME = new_game()   # helps for function

end
