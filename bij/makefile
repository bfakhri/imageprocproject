CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: sift surf

sift: SIFT.cpp
	g++ SIFT.cpp -o SIFT.exec -g $(CFLAGS) $(LIBS) 

surf: SURF.cpp
	g++ SURF.cpp -o SURF.exec -g $(CFLAGS) $(LIBS) 
