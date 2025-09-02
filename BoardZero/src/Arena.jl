module Arena

export MCTS_vs_MCTS_arena, AZ_vs_AZ_arena, AZ_vs_AZ_match

using Random
using DataFrames
using ProgressMeter
using Flux, AMDGPU, GPUArrays
using BSON
using Distributions: Categorical


include("params.jl")

using ..Game
using ..Algorithm
using ..MCTSmodule, ..AZmodule , ..ParallelAZ
using ..NNet, ..Stats

include("Arena/elo.jl")
using .ELO


# Perform single match
include("Arena/PvP.jl")
export human_vs_AZ, human_vs_MCTS, MCTS_vs_AZ, human_vs_human, MCTS_vs_MCTS, AZ_vs_AZ, moves_helper

# Get elo thanks to tournaments
include("Arena/multi.jl")
export MCTS_arena, AZ_vs_MCTS_arena, AZ_arena

end

