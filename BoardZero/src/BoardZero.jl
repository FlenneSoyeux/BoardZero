module BoardZero

include("params.jl")

include("Games/Game.jl")
using .Game

include("Games/NNet.jl")
using .NNet

include("Algorithm/Algorithm.jl")
using .Algorithm

include("Algorithm/AZ/Stats.jl")
using .Stats

include("Algorithm/AZ/ParallelAZ.jl")

include("Arena.jl")
using .Arena

include("Algorithm/AZ/Trainer.jl")
using .Trainer

using Random
using Flux, AMDGPU
using GPUArrays
using BSON


function __init__()

#     MCTS_arena()

    #Stats.print()

    #MCTS_vs_MCTS_arena([400], NGAMES=128)
    #MCTS_vs_MCTS(10000, 0.0)
#      MCTS_vs_AZ(0, 5.0, 0, 5.0)
       #human_vs_AZ(400, 0.0)
#       human_vs_MCTS(100000, 0.0)
    #moves_helper(0, 5.0; newGame=true, useMCTS=false)
    #AZ_vs_AZ(100, 0.0)
        # test()
#         Arena.testarena()
       #human_vs_human()
#      Arena.evaluate_mcts()    # Evaluate all MCTS written in PARAMS_MCTS_ELO of params.jl

    #AZ_arena([6, 7, 8, 9]; NGAMES=100)   # Get ELO in matches vs previous assessed models
    #AZ_arena([3, 4, 5]; NGAMES=100)   # Get ELO in matches vs previous assessed models

    
    #Arena.AZ_vs_MCTS_arena()      # Get ELO in matches vs classical MCTS
    #Arena.set_first_to_zero()
    #Model.save_onnx()         #Save to ONNX (for website)
    Trainer.train_parallel()
    #Trainer.learn_from_scratch()

#     nna = Model.initialize_model(39)
#     nnb = Model.initialize_model("Santorini_1600/model_39.bson")
#     res = Arena.AZ_vs_AZ_match(nna, nnb, 200)
#     Arena.AZ_vs_AZ(0, 0.25, :last, :last)

    #nna = Model.initialize_model(:ELO)
    #nnb = Model.initialize_model(:ELO)
    #res = Arena.AZ_vs_AZ_match(nna, nnb, 200)
    #println("res = ", res)
    #AZ_vs_AZ(1600, 0.0, 11, 10)

#     println("res = ", res)
#=
        for id in 2:2
            dict = BSON.load(DIRECTORY*"stats_"*string(id)*".bson")
        dict[:ELO] = 1000
        BSON.bson(DIRECTORY*"stats_"*string(id)*".bson", dict)
        end=#

     #=nn = NNet.initialize_model(:ELO)
     w = Arena.AZ_vs_AZ_match(nn, nn; NGAMES=200)
     println("w for id1 = ", 100*w, " %")=#

    #Stats.print()

    #Trainer.transfert()

    

end


end
