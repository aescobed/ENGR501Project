#ifndef C_COMMUNICATOR_H
#define C_COMMUNICATOR_H

void* StartClient();

void EndClient(void* client);

struct tensorSender {
    int size;
    void* client;
    double *tensor;
};

struct tensorSender CreateTensorSender(int size, void* client);

void DeleteTensorSender(tensorSender ts);

#endif