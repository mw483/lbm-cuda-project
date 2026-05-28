# Makefile
CC 			= g++
#CFLAGS 	= -O3 -fopenmp -Wall -g
CFLAGS  = -O3 -Wall -g
LDFLAGS =
LIBS 		=

TARGET 	= run_analysis
SRCS 		= main.cpp
include SourceList.mak

OBJS    = $(filter %.o,$(SRCS:%.cpp=%.o))

%.o:%.cpp
					$(CC) $(CFLAGS) -c $^

all:$(TARGET)

$(TARGET):$(OBJS)
					$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f *.o
	rm -f $(TARGET)
