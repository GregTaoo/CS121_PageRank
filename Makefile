CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -g -MMD -MP -fopenmp
LDFLAGS ?= -fopenmp
TARGET ?= pagerank
BUILD_DIR := build

SRCS := $(wildcard *.c)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean distclean rebuild

all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d

distclean: clean
	rm -rf $(BUILD_DIR)/$(TARGET)

rebuild: distclean all