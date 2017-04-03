CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: sift2 sift6 surf2 surf6

sift2: SIFT_2-6.cpp
	g++ SIFT_2-6.cpp -o SIFT_2-6.exec -g $(CFLAGS) $(LIBS) 

sift6: SIFT_6-2.cpp
	g++ SIFT_6-2.cpp -o SIFT_6-2.exec -g $(CFLAGS) $(LIBS) 

surf2: SURF_2-6.cpp
	g++ SURF_2-6.cpp -o SURF_2-6.exec -g $(CFLAGS) $(LIBS) 

surf6: SURF_6-2.cpp
	g++ SURF_6-2.cpp -o SURF_6-2.exec -g $(CFLAGS) $(LIBS) 
