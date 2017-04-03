CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: sift26 sift62 surf26 surf62

sift26: SIFT_2-6.cpp
	g++ SIFT_2-6.cpp -o SIFT_2-6.exec -g $(CFLAGS) $(LIBS) 

sift62: SIFT_6-2.cpp
	g++ SIFT_6-2.cpp -o SIFT_6-2.exec -g $(CFLAGS) $(LIBS) 

surf26: SURF_2-6.cpp
	g++ SURF_2-6.cpp -o SURF_2-6.exec -g $(CFLAGS) $(LIBS) 

surf62: SURF_6-2.cpp
	g++ SURF_6-2.cpp -o SURF_6-2.exec -g $(CFLAGS) $(LIBS) 
