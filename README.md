# BoardZero

BoardZero is an implementation of the AlphaZero algorithm giving an IA for some non obvious games. It is written in Julia and is using the AMDGPU.jl library, meaning that it'll run on AMD GPUs. 

## How to install ?
- You'll need to have Julia installed
- Download this git repository and then simply run in a terminal :
```
julia -t 12 --gcthreads 4,1 --project="BoardZero" BoardZero/src/BoardZero.jl Azul
```
The \__init__() function that is in the main file, namely /BoardZero/src/BoardZero.jl, will launch with the game "Azul"

## How to use ?
Instead of Azul you can launch the project with the other games. Available : `Santorini`, `Azul`, `Boop`, `Resolve`.


Depending on what you want to do, you'll need to comment/uncomment lines in BoardZero.jl.
