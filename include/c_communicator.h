#ifndef C_COMMUNICATOR_H
#define C_COMMUNICATOR_H

void* StartClient();

void EndClient(void* client);

struct tensorSender {
    int size;
    const char* tensor_key;
    void* client;
    double *tensor;
    size_t key_length;
    size_t dims[2];
};

struct tensorSender CreateTensorSender(int size, void* client);

void DeleteTensorSender(tensorSender* ts);

void SetTensorParameterValue(struct tensorSender* ts, int ind, double val);

void SetTensorOutputValue(struct tensorSender* ts, int ind, double val);

void SendTensor(struct tensorSender* ts);

void* GetTensor(struct tensorSender* ts);

void ReceiveAndPrintTensorSender(struct tensorSender* ts);

#endif