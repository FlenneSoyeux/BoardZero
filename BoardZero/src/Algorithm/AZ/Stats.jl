module Stats

using BSON, DataFrames, Plots

using ..Algorithm, ..Game, ..AZmodule, ..NNet
include("../../params.jl")


"""
A Stat element is produced at the start of the training. It gathers data (mean loss, game played, average rounds, etc...) until it is saved, where another Stat element is made.
Stats are saved in a file and then read and plotted to track evolution.
"""

mutable struct Stat
    id::Int64
    ep::Int64
    time::Float64
    games::Int64
    ELO::Int64
    loss::Float32
    loss_pi::Float32
    loss_v::Float32
    loss_points::Float32
    average_round::Float32
    AZ_depth::Int64
    NNname::String
    Stat() = new(1, 0, 0, 0, -1, -1, -1, -1, -1, -1, LEARNING_PARAMS["MCTS_ITER"], "NoName")
    Stat(S::Stat) = new(S.id+1, S.ep, S.time, S.games, -1, S.loss, S.loss_pi, S.loss_v, S.loss_points, S.average_round, LEARNING_PARAMS["MCTS_ITER"], S.NNname)
    Stat(d::Dict) = new(d[:id], d[:ep], d[:time], d[:games], round(Int, d[:ELO]), d[:loss],
                        haskey(d, :loss_pi) ? d[:loss_pi] : -1,
                        haskey(d, :loss_v) ? d[:loss_v] : -1,
                        haskey(d, :loss_points) ? d[:loss_points] : -1,
                        haskey(d, :average_round) ? d[:average_round] : -1,
                        haskey(d, :AZ_depth) ? d[:AZ_depth] : -1, 
                        haskey(d, :NNname) ? d[:NNname] : "NoName")
end


function update!(S::Stat, dict)
    S.ep += dict[:ep]
    S.time += dict[:time]
    S.games += dict[:games]
    S.loss = dict[:loss]
    S.loss_pi = dict[:loss_pi]
    S.loss_v = dict[:loss_v]
    S.loss_points = dict[:loss_points]
    S.average_round = dict[:average_round]
    S.NNname = dict[:NNname]
end


function save_stats(S::Stat)
    BSON.bson(DIRECTORY*"stats_"*string(S.id)*".bson", Dict(:id=>S.id, :ep=>S.ep, :time=>S.time, :games=>S.games, :ELO=>S.ELO, :loss=>S.loss, :loss_pi=>S.loss_pi, :loss_v=>S.loss_v, :loss_points=>S.loss_points, :average_round=>S.average_round, :AZ_depth=>S.AZ_depth, :NNname=>S.NNname))
end

function print(path = DIRECTORY)
    id = 1
    all_stats = Stats.Stat[]
    while true
        S = load_stats(id, path)
        !isnothing(S) || break
        push!(all_stats, S)
        id += 1
    end
    Stats.print(all_stats)
end

function print(V::Vector{Stat})
    df = DataFrame(V)
    display(df)

    if length(V) == 0
        return
    end

    #Plots
    ELO = [S.ELO for S in V]
    time = [S.time/3600 for S in V]
    games = [S.games for S in V]
    loss = [S.loss for S in V]
    loss_v = [S.loss_v for S in V]
    loss_pi = [S.loss_pi for S in V]
    average_round = [S.average_round for S in V]

    #timeticks = [0:3600:time[end]..., time[end]]
    eloticks = [0:100:maximum(ELO)..., maximum(ELO)]

    #ELO
    Plots.plot(time, ELO, xlims=[0,Inf],ylims=[0,Inf], yticks=eloticks); Plots.savefig(DIRECTORY*"elo_vs_time.png")
    Plots.plot(games, ELO, xlims=[0,Inf],ylims=[0,Inf], yticks=eloticks); Plots.savefig(DIRECTORY*"elo_vs_games.png")

    #loss
    Plots.plot(time, [loss, loss_pi, loss_v], xlims=[0,Inf],ylims=[0,Inf]); Plots.savefig(DIRECTORY*"loss_vs_time.png")   #loss vs time
    Plots.plot(games, [loss, loss_pi, loss_v], xlims=[0,Inf],ylims=[0,Inf]); Plots.savefig(DIRECTORY*"loss_vs_games.png")   #loss vs games

    #average_round
    Plots.plot(time, average_round); Plots.savefig(DIRECTORY*"rounds_vs_time.png")   #rounds vs time
    Plots.plot(games, average_round); Plots.savefig(DIRECTORY*"rounds_vs_games.png")   #rounds vs games

    Plots.closeall()
end

function load_stats(id::Int64, path = DIRECTORY)
    #return Stats
    filename = path*"stats_"*string(id)*".bson"
    if !isfile(filename)
        return nothing
    end
    d = BSON.load(filename)
    return Stat(d)
end

end
