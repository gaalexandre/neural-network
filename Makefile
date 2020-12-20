CC = g++
CFLAGS = -Wall -std=c++11
EXEC=nn

OBJS= \
NeuralNetwork.o \
fonctions.o \
Aprentissage.o \
main.o \


$(EXEC):$(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lpthread

%.o:%.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I/usr/include/eigen3/

clean:
	rm *.o $(EXEC)
