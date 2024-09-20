################################################################################
#                                                                              #
#   SFEM   ++                                                                  #
#                                                                              #
#   created April '18                                                          #
#                                                                              #
################################################################################
#
CPP = clang++ -fPIC -std=c++11
INCLUDE = /usr/local/include/eigen3 -I. -I..
CPPFLAGS = -O3
LDFLAGS = -lm -lblas -llapack
VPATH = SparseQuadrature
OBJECTS = SparseQuadrature.o TDindexSet.o HaltonSet.o\
          TensorProductQuadrature.o GaussLegendreQuadratureProb.o

all: example_HeatEquationUQ.o $(OBJECTS)
	$(CPP) example_HeatEquationUQ.o $(OBJECTS) -o a.out $(LDFLAGS)  

#
# tell make how to create a .o from a .c
#

%.o:%.cpp
	$(CPP)  $(CPPFLAGS) -I$(INCLUDE) -c $<
#
.PHONY: clean
#
clean:
	rm -f *.o *.out

#
# DO NOT DELETE THIS LINE -- make depend depends on it.
