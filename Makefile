CXX      = g++
CXXFLAGS = -O3 -std=c++17 -march=native -Wall -Wextra -fopenmp

.PHONY: all solver cli sim solver_misere sim_misere clean

all: solver cli sim solver_misere sim_misere

solver: src/booty_dice_solver.cpp
	$(CXX) $(CXXFLAGS) -o solver src/booty_dice_solver.cpp

cli: src/booty_dice_cli.cpp
	$(CXX) $(CXXFLAGS) -o cli src/booty_dice_cli.cpp

sim: src/booty_dice_sim.cpp
	$(CXX) $(CXXFLAGS) -o sim src/booty_dice_sim.cpp

solver_misere: src/booty_dice_solver_misere.cpp
	$(CXX) $(CXXFLAGS) -o solver_misere src/booty_dice_solver_misere.cpp

sim_misere: src/booty_dice_sim_misere.cpp
	$(CXX) $(CXXFLAGS) -o sim_misere src/booty_dice_sim_misere.cpp

clean:
	rm -f solver cli sim solver_misere sim_misere
