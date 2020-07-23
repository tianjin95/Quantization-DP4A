CC = gcc
CFLAGS = -g3 -o0

all:infer.c
	$(CC) $(CFLAGS) infer.c -o main -g -lm

clean:
	rm *.o main
