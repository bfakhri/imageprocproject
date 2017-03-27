CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

sift: SIFT.cpp
	g++ SIFT.cpp -o SIFT.exec -g $(CFLAGS) $(LIBS) 
