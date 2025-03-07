.PHONY: all clean

CXXFLAGS=-O3 -ffast-math -march=native -shared -fPIC
CXXFLAGS_MAIN=-O3 -ffast-math -march=native 
INCLUDES=-I./src -I/usr/local/include -I/opt/homebrew/include  # path may need to be adjusted here !
LINKS=-L/usr/local/lib -L/opt/homebrew/lib
LIBS=-lm -lgsl -lcblas  # libs may need to be adjusted here!

# determine source, header, and object files
SRC=src/localGt+.cpp
HDRS=src/localGt+.h
OBJ=$(patsubst src/%, lib%, $(SRC:.cpp=.so))
CYTHSRC=$(wildcard cpot.p*)
CTHOUT=cpot.cpp

all : $(OBJ) $(CTHOUT)

$(CTHOUT) : $(CYTHSRC) $(SRC) $(HDRS) setup.py
	python3 setup.py build_ext --inplace

lib%.so : src/%.cpp src/%.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LINKS) $(LIBS) -c $< -o $@

test: all
	python3 main.py -rm 25 -p chiral -er 0.01 40. 60 -lmax 4

clean :
	@rm -f $(OBJ) *.so cpot.cpp

BACKUP:=backup_evc_`date +"%Y-%m-%d"`.zip
backup :
	@git archive --output=$(BACKUP) --prefix="backup_" HEAD
	@rsync -a --progress $(BACKUP) $(FRIB_BACKUP_FOLDER)
	@rm $(BACKUP)

lec_output: $(SRC)
	$(CXX) $(CXXFLAGS_MAIN) $(INCLUDES) $(LINKS) $(LIBS) -DUNITTEST=1 -o $@ $<