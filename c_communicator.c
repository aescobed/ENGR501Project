
#include "c_client.h"
#include "c_configoptions.h"
#include <stdio.h>
#include "include/c_communicator.h"

using namespace SmartRedis;

void* StartClient()
{
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
        return NULL;
    }

    err = CreateClient(config_options, logger_name, cid_len, &client);
    if (err != SRNoError) {
        printf("Error creating SmartRedis client!\n");
        return NULL;
    }

    return client;
}

void EndClient(void* client)
{
    DeleteCClient(&client);
}


struct tensorSender CreateTensorSender(int size, void* client)
{
    struct tensorSender ts;
    
    ts.size = size;
    ts.client = client;
    ts.tensor = (double*)malloc(sizeof(double) * size);

    return ts;
}


void DeleteTensorSender(tensorSender ts)
{
    free(ts.tensor);
}
