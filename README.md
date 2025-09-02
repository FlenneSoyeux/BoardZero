# BoardZero

BoardZero is an implementation of the AlphaZero algorithm giving an IA for some non obvious games. It is written in Julia and is using the AMDGPU.jl library, meaning that it'll run on AMD GPUs. 

## How to install ?
- You'll need to have Julia installed
- Download this git repository and then simply run in a terminal :
```
julia --project="BoardZero"
using BoardZero
```
The \__init__() function that is in the main file, namely /BoardZero/src/BoardZero.jl, will launch. 

## How to use ?
First, you'll have to edit the `params.jl`, the `GAME = ` line to choose the game you want to have. You can write `"Santorini"` if you want to play the Santorini game. Available : `"Santorini"`, `"Azul"`, `"Boop"`, `"Resolve"`.

Depending on what you want to do, you'll need to comment/uncomment lines in BoardZero.jl. The most useful ones :
- `moves_helper(0, 5.0; newGame=true, useMCTS=false)` to run a game move by move and see what the AlphaZero algorithm suggests. If `newGame` is false you can edit a position and start from there. The first 2 parameters are : maximal number of iterations (if not 0) and maximal seconds of search (if not 0.0).
- `AZ_vs_AZ(1000, 0.0)` to see a game between AlphaZero and itself. Set the maximal number of iterations (or 0) or the maximal time of search in seconds.
- `human_vs_AZ(400, 0.0)` to play against AlphaZero with same parameters as above.
- `Trainer.train_parallel()` will start a routine to train the neural network and make it progress.
