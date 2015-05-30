SHELL		=	/bin/bash
CXX		=	g++
CXXFLAGS	=	-O3 -fomit-frame-pointer -finline-functions -I../include -I../tests/include
LIBS		=	-lclblas -lOpenCL -lboost_program_options -lrt
.PHONY: clean

%.o:%.cpp
	${CXX} ${CXXFLAGS} $< -c

clblas_client: clblas_client.o statisticalTimer.o timer.o
	${CXX} ${CXXFLAGS} $^ ${LIBS} -o $@ 

clean:
	rm -rf *.o
