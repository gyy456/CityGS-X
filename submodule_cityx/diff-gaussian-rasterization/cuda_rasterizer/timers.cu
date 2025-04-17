#include <chrono>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include "stdio.h"
#include "stdlib.h"

class MyTimerOnGPU {
public:
    void start(const std::string& name) {
        cudaEventCreate(&start_events[name]);
        cudaEventRecord(start_events[name]);
        durations[name] = -1;
    }

    void stop(const std::string& name) {
        cudaEventCreate(&stop_events[name]);
        cudaEventRecord(stop_events[name]);
    }

    double elapsedMilliseconds(const std::string& name, const std::string& mode = "average") {
        if (durations[name] > -0.5) {
            return durations[name];
        }
        cudaEventSynchronize(stop_events.at(name));
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_events.at(name), stop_events.at(name));
        durations[name] = (double)milliseconds;
        return (double)milliseconds;
    }

    void printAllTimes(
        int iteration,
        int world_size,
        int local_rank,
        std::string log_folder,
        bool print_prefix
    ) {
        char* prefix = new char[64];
		sprintf(prefix, "\nit=%d,ws:%d,rk=%d  -->\n", iteration, world_size, local_rank);
		char* filename = new char[64];
		sprintf(filename, "%s/gpu_time_ws=%d_rk=%d.log", log_folder.c_str(), world_size, local_rank);
        // merge the above two lines into one line

        std::vector<std::pair<std::string, double>> sortedTimes(durations.begin(), durations.end());
        std::sort(sortedTimes.begin(), sortedTimes.end(), 
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );
        // if (print_prefix) {
        //     std::cout << prefix << std::endl;
        // }
        // for (const auto& pair : sortedTimes) {
        //     std::cout << pair.first << " time: " << elapsedMilliseconds(pair.first, "sum") << " ms" << std::endl;
        // }
        //save in file
        FILE *fp;
        fp = fopen(filename, "a");
        if (print_prefix) {
            fprintf(fp, "%s", prefix);
        }
        for (const auto& pair : sortedTimes) {
            fprintf(fp, "%s time: %f ms\n", pair.first.c_str(), elapsedMilliseconds(pair.first, "sum"));
        }
        //clean up
        fclose(fp);
        delete[] prefix;
        delete[] filename;
    }

protected:
    std::unordered_map<std::string, double> durations;
    std::unordered_map<std::string, cudaEvent_t> start_events;
    std::unordered_map<std::string, cudaEvent_t> stop_events;
};