CC=g++  -g -std=c++11

INC=$(shell pkg-config --cflags opencv)
LIB=$(shell pkg-config --libs opencv)
main:main.o fun.o 
	$(CC) -o main main.o fun.o  $(LIB)
main.o:main.cpp main.h 
	$(CC) -c $< $(INC) -o $@
fun.o:func.cpp func.h 
	$(CC) -c $< $(INC) -o $@

.PHONY:clean
clean:
	rm *.o
