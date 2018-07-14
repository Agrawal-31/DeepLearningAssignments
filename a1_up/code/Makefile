CC = g++
CFLAGS = -std=c++14 -Wall -O2

EXEC_PATH = aout
SRC_DIR = src
OBJ_DIR = obj

OBJ_FILES = $(addprefix $(OBJ_DIR)/, classifiers.o utilities.o main.o)

.PHONY: all clean

all: $(OBJ_FILES)
	$(CC) -o $(EXEC_PATH) $(OBJ_FILES) $(CFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/*.h
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -f $(OBJ_DIR)/*