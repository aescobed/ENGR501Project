#include "c_client.h"
#include "c_configoptions.h"
#include <stdio.h>

using namespace SmartRedis;

int main(int argc, char* argv[]) {

    SRError err;

    // Initialize SmartRedis client
    void* client = NULL;
    // Assuming `config_options` and `logger_name` are set up correctly
    // For simplicity, these are set to NULL and "" in this example
    const char* logger_name = "send_tens";
    const size_t cid_len = strlen(logger_name);

    void* config_options = NULL;

    const char* db_suffix = "OPTIMIZER";
    const size_t db_suffix_len = strlen(db_suffix);

    // Uses the suffix from the SSDB environment variable
    err = create_configoptions_from_environment(db_suffix, db_suffix_len, &config_options);
    if (err != SRNoError || config_options == NULL) {
        printf("Error creating config options.\n");
        return -1;
    }





    err = CreateClient(config_options, logger_name, cid_len, &client);
    if (err != SRNoError) {
        printf("Error creating SmartRedis client!\n");
        return -1;
    }

    // Example tensor data and dimensions
    double tensor_data[4] = {1.0, 2.0, 3.0, 4.0};
    size_t dims[1] = {4};

    // Put tensor into the database
    const char* tensor_key = "my_tensor";
    size_t key_length = strlen(tensor_key);

    put_tensor(client, tensor_key, key_length, tensor_data, dims, 1, SRTensorTypeDouble, SRMemLayoutNested);

    printf("Tensor put into database: %s\n", tensor_key);

    // Always delete the client when done
    DeleteCClient(&client);

    return 0;
}
