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

    """    
    Uncomment the function calls you want to execute.
    """
    
    Stats.print()   # List all the models, strength, etc.. The next functions will always take the strongest or the last one :
    
    # ┌─────────────────────────────────────────────────────────────┐
    # │                    PLAY WITH OR AGAINST THE IA              │
    # └─────────────────────────────────────────────────────────────┘
    
    #human_vs_human()                       # Enter each moves manually
    #human_vs_AZ(maxIteration=1000)                  # You play vs the IA ! Be careful it's strong
    #human_vs_MCTS(100000, 0.0)             # If you want to play the vanilla MCTS algorithm
    #AZ_vs_AZ(timeLimit=10.0)                     # Show match of the IA against itself 
    moves_helper(timeLimit=5.0, newGame=true)    # Game diagnosis : enter each moves you want AND have advice from the IA
    
    # ┌─────────────────────────────────────────────────────────────┐
    # │                     TRAIN THE IA                            │
    # └─────────────────────────────────────────────────────────────┘
    
    
    #Trainer.train_parallel()               # Open previous NN (or create a blank new, see options in this function), runs several games, learns from it, repeat
    #Trainer.learn_from_scratch()           # Create a blank new NN and learns only from the stored positions (replayBuffer.jls)

    #AZ_arena([5, 18, 19]; NGAMES=100)      # Makes all the possible matches between the given model numbers, with NGAMES at every time, and compute an ELO from it
    #AZ_arena([5, 19, 20]; NGAMES=100)      # Makes all the possible matches between the given model numbers, with NGAMES at every time, and compute an ELO from it

    
    # ┌─────────────────────────────────────────────────────────────┐
    # │                     OTHER FUNCTIONS (useless)               │
    # └─────────────────────────────────────────────────────────────┘


    #### Other functions :
    #MCTS_arena()                               # Compute ELOs of vanilla MCTS algorithms with several given iterations (iterations written in params.jl)
    #MCTS_vs_MCTS_arena([400], NGAMES=128)      # Do MCTS1 vs MCTS2. You can write options for MCTS1 for instance. It helps studying vanilla MCTS.
    #MCTS_vs_AZ(0, 5.0, 0, 5.0)                 # Evaluate MCTS vs AZ algorithm
    #Arena.AZ_vs_MCTS_arena()                   # Get ELO in matches vs classical MCTS
    #Model.save_onnx()                          # Save to ONNX (for website) - not working now

    ## NGAMES matches of nna vs nnb to see who wins 
    #=nna = Model.initialize_model(:ELO)
    nnb = Model.initialize_model(1)
    w = Arena.AZ_vs_AZ_match(nna, nnb, 200)
    println("Winrate for the first model : ", 100*w, " %")=#


end


end
