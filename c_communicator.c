
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
    ts.tensor = (double*)malloc(sizeof(double) * size * 2);

    if (ts.tensor == NULL) {
        fprintf(stderr, "Failed to allocate memory for the tensor.\n");
        exit(EXIT_FAILURE);
    }

    ts.tensor_key = "parameters";
    ts.dims[0] = 2;
    ts.dims[1] = size;
    ts.key_length = strlen(ts.tensor_key);

    return ts;
}



void DeleteTensorSender(tensorSender* ts)
{
    free(ts->tensor);
}



void SetTensorParameterValue(struct tensorSender* ts, int ind, double val)
{
    if (ind < 0 || ind >= ts->size)

        printf("Error Setting tensor!\n");

    else
        ts->tensor[ind] = val;

}


void SetTensorOutputValue(struct tensorSender* ts, int ind, double val)
{
    if (ind < 0 || ind >= ts->size)

        printf("Error Setting tensor!\n");

    else
        ts->tensor[ts->size + ind] = val;

}



void SendTensor(struct tensorSender* ts)
{
    put_tensor(ts->client, ts->tensor_key, ts->key_length, ts->tensor, ts->dims, 2, SRTensorTypeDouble, SRMemLayoutContiguous);
}

void* GetTensor(struct tensorSender* ts)
{
    void* tensor_receive_data = NULL;
    size_t* dims_receive = NULL;
    size_t num_dims_receive;

    SRTensorType TensorTypeReceived;

    get_tensor(ts->client, ts->tensor_key, ts->key_length, &tensor_receive_data, &dims_receive, &num_dims_receive, &TensorTypeReceived, SRMemLayoutNested);

    return tensor_receive_data;
}


void ReceiveAndPrintTensorSender(struct tensorSender* ts)
{
    void* tensor_receive_data = NULL;

    tensor_receive_data = GetTensor(ts);

    printf("%s={{", ts->tensor_key);

    double* data = (double*)tensor_receive_data;

    int i;
    for(i = 0; i < ts->dims[1]; i++)
    {
        printf("%f", data[i]);
        if(i<ts->dims[1])
            printf(",");
    }

    printf("},\n\t\t{");

    for(i = ts->dims[0]; i < ts->dims[0] + ts->dims[1]; i++)
    {
        printf("%f", data[i]);
        if(i<ts->dims[1])
            printf(",");
    }

    printf("}}\n");
}