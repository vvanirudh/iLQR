all:
	g++ --std=c++11 ilqr_diffdrive.cpp main.cpp -I/usr/include/eigen3 -o diffdrive	
