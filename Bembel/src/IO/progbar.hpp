
#ifndef BEMBEL_IO_PROGBAR_H_
#define BEMBEL_IO_PROGBAR_H_

namespace Bembel {
namespace IO {
class progressbar {
 public:
  // default destructor
  ~progressbar() = default;

  // delete everything else
  progressbar(progressbar const&) = delete;
  progressbar& operator=(progressbar const&) = delete;
  progressbar(progressbar&&) = delete;
  progressbar& operator=(progressbar&&) = delete;

  // default constructor, must call set_niter later
  progressbar();
  progressbar(int n);

  // reset bar to use it again
  void reset();
  // set number of loop iterations
  void set_niter(int iter);

  // main function
  void update();

 private:
  int progress;
  int n_cycles;
  int last_perc;
  bool update_is_called;

  std::string done_char;
  std::string todo_char;
  std::string opening_bracket_char;
  std::string closing_bracket_char;

  std::chrono::steady_clock::time_point start_time;
};

progressbar::progressbar()
    : progress(0),
      n_cycles(0),
      last_perc(0),
      update_is_called(false),
      done_char("#"),
      todo_char(" "),
      opening_bracket_char("["),
      closing_bracket_char("]"),
      start_time(std::chrono::steady_clock::now()) {}

progressbar::progressbar(int n)
    : progress(0),
      n_cycles(n),
      last_perc(0),
      update_is_called(false),
      done_char("#"),
      todo_char(" "),
      opening_bracket_char("["),
      closing_bracket_char("]"),
      start_time(std::chrono::steady_clock::now()) {
  std::cout << "BAR INFO: #: 2%, elapsed time / remaining time, iters/total_iters"
            << std::endl;
}

void progressbar::reset() {
  progress = 0, update_is_called = false;
  last_perc = 0;
  start_time = std::chrono::steady_clock::now();
  return;
}

void progressbar::set_niter(int niter) {
  if (niter <= 0)
    throw std::invalid_argument(
        "progressbar::set_niter: number of iterations null or negative");
  n_cycles = niter;
  return;
}

void progressbar::update() {
  if (n_cycles == 0)
    throw std::runtime_error("progressbar::update: number of cycles not set");

  if (!update_is_called) {
    std::cout << opening_bracket_char;
    for (int _ = 0; _ < 50; _++) std::cout << todo_char;
    std::cout << closing_bracket_char << " --:--:--/--:--:-- "
              << std::to_string(progress) << '/' << std::to_string(n_cycles);
  }
  update_is_called = true;

  int perc = 0;
  perc = progress * 100. / (n_cycles - 1);
  // compute percentage, if did not change, do nothing and return
  int n_cycles_size = std::to_string(n_cycles).size();
  int progress_size = std::to_string(progress).size();

  std::cout << std::string(n_cycles_size + progress_size + 2, '\b');
  // erase time
  std::cout << std::string(17, '\b');

  auto elapsed = (std::chrono::steady_clock::now() - start_time);
  auto remain = elapsed / (progress + 1) * (n_cycles - progress - 1);

  int elapsed_s =
      std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
  int elapsed_h = elapsed_s / 3600;
  int elapsed_m = (elapsed_s - (elapsed_h * 3600)) / 60;
  elapsed_s = elapsed_s - elapsed_m * 60 - elapsed_h * 3600;

  std::cout << std::setfill('0') << std::setw(2) << elapsed_h << ":";
  std::cout << std::setfill('0') << std::setw(2) << elapsed_m << ":";
  std::cout << std::setfill('0') << std::setw(2) << elapsed_s << "/";

  int remain_s =
      std::chrono::duration_cast<std::chrono::seconds>(remain).count();
  int remain_h = remain_s / 3600;
  int remain_m = (remain_s - (remain_h * 3600)) / 60;
  remain_s = remain_s - remain_m * 60 - remain_h * 3600;


  std::cout << std::setfill('0') << std::setw(2) << remain_h << ":";
  std::cout << std::setfill('0') << std::setw(2) << remain_m << ":";
  std::cout << std::setfill('0') << std::setw(2) << remain_s << " ";

  std::cout << std::to_string(progress + 1) << '/' << std::to_string(n_cycles);

  // update bar every ten units
  if (perc % 2 == 0) {
    // erase trailing percentage characters
    std::cout << std::string(
        n_cycles_size + std::to_string(progress + 1).size() + 2, '\b');
    // erase time
    std::cout << std::string(18, '\b');
    // erase closing bracket
    std::cout << std::string(closing_bracket_char.size(), '\b');

    // erase 'todo_char'
    for (int j = 0; j < 50 - (perc - 1) / 2; ++j) {
      std::cout << std::string(todo_char.size(), '\b');
    }

    // add one additional 'done_char'
    if (perc == 0)
      std::cout << todo_char;
    else
      std::cout << done_char;

    // refill with 'todo_char'
    for (int j = 0; j < 50 - (perc - 1) / 2 - 1; ++j) std::cout << todo_char;

    // readd trailing percentage characters
    std::cout << closing_bracket_char << " ";
    std::cout << std::setfill('0') << std::setw(2) << elapsed_h << ":";
    std::cout << std::setfill('0') << std::setw(2) << elapsed_m << ":";
    std::cout << std::setfill('0') << std::setw(2) << elapsed_s << "/";

    std::cout << std::setfill('0') << std::setw(2) << remain_h << ":";
    std::cout << std::setfill('0') << std::setw(2) << remain_m << ":";
    std::cout << std::setfill('0') << std::setw(2) << remain_s << " ";

    std::cout << std::to_string(progress + 1) << '/'
              << std::to_string(n_cycles);
  }
  last_perc = perc;
  ++progress;
  std::cout << std::flush;

  return;
}

}  // namespace IO
}  // namespace Bembel

#endif
