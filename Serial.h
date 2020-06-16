#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <Windows.h>


class Serial {
public:
    void read(std::string& answer);
    void write(const std::string& message);
    void query(const std::string& request, std::string& answer, int delay);
    std::string query(const std::string& request, int delay);

    explicit Serial(const std::string& port_name);
    ~Serial();
private:
    void test_serial();
    std::string port_name;
    HANDLE       comm_handle;
    DCB          dcb_params;
    COMMTIMEOUTS timeouts;
    int          QUERY_SLEEP_DURATION;
};
