EXECUTABLES = sig
CC=gcc

all: ${EXECUTABLES}

LDFLAGS += $(foreach librarydir,$(subst :, ,$(LD_LIBRARY_PATH)),-L$(librarydir))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  LDFLAGS += -lrt -lOpenCL -lm
  CFLAGS += -Wall -std=gnu99 -g -O2
endif
ifeq ($(UNAME_S),Darwin)
  LDFLAGS +=  -framework OpenCL -lm
  CFLAGS += -Wall -std=c99 -g -O2
endif

ifdef OPENCL_INC
  CPPFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  LDFLAGS = -L$(OPENCL_LIB)
endif

sigpr.o: sigpr.c cl-helper.h
cl-helper.o: cl-helper.c cl-helper.h

sig: sigpr.o cl-helper.o
	gcc -o ${EXECUTABLES} sigpr.o cl-helper.o ${LDFLAGS} ${CFLAGS}

clean:
	rm -f $(EXECUTABLES) *.o
