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
    size_t dims[1];
};

struct tensorSender CreateTensorSender(int size, void* client);

void DeleteTensorSender(tensorSender ts);

void SetTensorValue(struct tensorSender* ts, int ind, double val);

void SendTensor(struct tensorSender* ts);

void* GetTensor(struct tensorSender* ts);

#endif