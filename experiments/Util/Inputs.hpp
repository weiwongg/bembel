#ifndef inputs
#define inputs

#include <string.h>

#include <iostream>
#include <vector>

// should be equivalent to std::find, but for some reason it is not
template <class InputIterator, class T>
InputIterator findit(InputIterator first, InputIterator last, const T &val) {
  while (first != last) {
    if (*first == val) return first;
    ++first;
  }
  return last;
}

// source (author: iain):
// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class InputParser {
 public:
  InputParser(int &argc, char **argv) {
    for (int i = 1; i < argc; ++i) tokens_.push_back(std::string(argv[i]));
  }
  const std::string &getCmdOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = findit(tokens_.begin(), tokens_.end(), option);
    if (itr != tokens_.end() && ++itr != tokens_.end()) {
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }
  bool cmdOptionExists(const std::string &option) const {
    return findit(tokens_.begin(), tokens_.end(), option) != tokens_.end();
  }
  const std::string &getRequiredCmd(const std::string &option) const {
    if (!cmdOptionExists(option)) {
      std::cout << option << " is a required input option" << std::endl;
      exit(1);
    }
    return getCmdOption(option);
  }


  // shortcuts
  const std::string &getCmd(const std::string &option) const {
    return getCmdOption(option);
  }
  bool cmdExists(const std::string &option) const {
    return cmdOptionExists(option);
  }

 private:
  std::vector<std::string> tokens_;
};

#endif
