CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -g -MMD -MP -fopenmp
LDFLAGS ?= -fopenmp
TARGET ?= pagerank
BUILD_DIR := build

# 当前目录下所有源文件
SRCS := $(wildcard *.c)
# 对应的对象文件放在 build/ 目录
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean distclean rebuild

all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET)

# 确保 build 目录存在
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 链接可执行文件
$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

# 编译规则：.c -> build/%.o
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# 包含依赖文件
-include $(DEPS)

# 清理中间文件
clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d

# 完全清理，包括可执行文件
distclean: clean
	rm -rf $(BUILD_DIR)/$(TARGET)

# 强制重新构建
rebuild: distclean all