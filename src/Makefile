# Makefile
CUDA_VERSION = 8.0
MPI_VERSION  = 2.1.2


SRCS = main.cu
include SourceList_Particle.mak

include sourceList_calculation.mak
include sourceList_function.mak
include sourceList_param.mak
include sourceList_output.mak

include sourceList_solid.mak


OBJS =  $(filter %.o,$(SRCS:%.c=%.o))
OBJS += $(filter %.o,$(SRCS:%.cpp=%.o))
OBJS += $(filter %.o,$(SRCS:%.cu=%.o))

ifeq ($(USEGPU),1)
  OBJS += $(filter %.o,$(SRCGPUS:%.cu=%.o))
endif


#CUDA_PATH = /apps/t3/sles12sp2/cuda/${CUDA_VERSION}
#MPI_PATH  = /apps/t3/sles12sp2/free/openmpi/${MPI_VERSION}/gnu
CUDA_PATH = /opt/cuda-8.0
#MPI_PATH = /home/inagaki/openmpi
MPI_PATH = /usr/local/openmpi-2.1.2

MPICC = mpic++
CC	  = mpicc
CXX	  = mpic++
NVCC  = nvcc


#tsubame
CFLAGS	  = -Wall -O3 -Wextra
CFLAGS   += -I${MPI_PATH}/include
CFLAGS   += -I${CUDA_PATH}/include

NVCCFLAGS  = -O3 -use_fast_math --std c++11
#NVCCFLAGS += -Xptxas -v
NVCCFLAGS += -I${MPI_PATH}/include
NVCCFLAGS += -I${CUDA_PATH}/include
NVCCFLAGS +=  --gpu-architecture=sm_60
NVCCFLAGS +=  -maxrregcount 63


LDFLAGS   = -lstdc++  -use_fast_math
LDFLAGS  += -L${MPI_PATH}/lib  -lmpi -lmpi_cxx
LDFLAGS  += -L${CUDA_PATH}/lib64  -lcudart
LDFLAGS  +=  --gpu-architecture=sm_60


# run
TARGET	=	run

%.o: %.c
		$(CC) $(CFLAGS) -c $<

%.o: %.cpp
		$(CXX) $(CFLAGS) -c $<

%.o: %.cu
		$(NVCC) $(NVCCFLAGS) -c $<


all: $(TARGET)

$(TARGET): $(OBJS)
		$(NVCC) $(LDFLAGS) -o $@ $(OBJS)
#$(TARGET): $(OBJS)
#		$(MPICC) $(LDFLAGS) -o $@ $(OBJS)

#etc
RRM     = *.o run_*.exe* *.oo *.linkinfo
RRMF    = .*.swp Makefile~ *.*~
RRMR    = ../result*/*.dat  ../result*/*.vtk  ../result*/*.vtr  ../result*/*.pvtr  ../result*/*.vtp ../result*/*.vtm

clean:
		rm -rf $(RRM) $(TARGET)

c :
		rm -rf $(RRM) $(TARGET)

resultclean :
		rm -rf $(RRM) $(RRMF) $(TARGET) $(RRMR)

depend :
		makedepend -Y -f depend.mk -- $(CFLAGS) -- $(SRCS)

-include depend.mk

# DO NOT DELETE
