isa:  HCCBINDIR=/opt/rocm/hcc-lc/bin
hsail:  HCCBINDIR=/opt/rocm/hcc-hsail/bin

isa: matmul
hsail: matmul

CXX = $(HCCBINDIR)/hcc
CXXFLAGS = $(shell $(HCCBINDIR)/hcc-config --cxxflags)
LDFLAGS = $(shell $(HCCBINDIR)/hcc-config --ldflags)

OBJECTS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
DEPS =  $(patsubst %.o,%.d,$(OBJECTS))

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
	$(CXX) -MM $(CXXFLAGS) $*.cpp -o $*.d

tile_static: $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

clean:
	rm -f matmul *.o *.d *~

-include $(DEPS)
