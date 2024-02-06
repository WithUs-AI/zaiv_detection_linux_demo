CC=g++
CFLAGS= -std=c++17 -lstdc++fs -Wall -lpthread  -I$(shell pwd)/src
LDFLAGS= -lhailort

EXECUTABLE=detection_native

SOURCES = $(wildcard *.cpp) $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

PKGNAMES := RapidJSON 

PKG_NAME_CHECK := opencv4
PKG_CHECK := $(shell pkg-config --exists $(PKG_NAME_CHECK) && echo 1 || echo 0)

ifeq ($(PKG_CHECK), 1)
	PKGNAMES += $(PKG_NAME_CHECK)
else
	PKG_NAME_CHECK := opencv
	PKG_CHECK := $(shell pkg-config --exists $(PKG_NAME_CHECK) && echo 1 || echo 0)
	ifeq ($(PKG_CHECK), 1)
		PKGNAMES += $(PKG_NAME_CHECK)
	endif
endif

CFLAGS += `pkg-config --cflags $(PKGNAMES)`
LDFLAGS += `pkg-config --libs $(PKGNAMES)`

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $(EXECUTABLE) $(OBJECTS) $(CFLAGS) $(LDFLAGS)
	
.cpp.o:
	$(CC) $(CFLAGS) $(LDFLAGS) $< -c -o $@
	
clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
	
