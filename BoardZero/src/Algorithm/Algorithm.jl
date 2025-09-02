module Algorithm


include("../params.jl")

using ..Game, ..NNet
using Random

abstract type AbstractAlgorithm end

# ALL functions an abstract algorithm should do

# A function returning a move to play
function get_move(A::AbstractAlgorithm, G::AbstractGame, rng = Random.default_rng()) :: AbstractMove end

function keep_subtree!(A::AbstractAlgorithm, m::AbstractMove) :: Nothing end



export AbstractAlgorithm, get_move, keep_subtree!, reset!

include("MCTS.jl")
using .MCTSmodule
export MCTSmodule, MCTSAlgorithm

include("MCTS_AMAF.jl")
using .AMAFmodule
export AMAFmodule, AMAFAlgorithm

include("AlphaZero.jl")
using .AZmodule
export AZmodule, AZAlgorithm

end
