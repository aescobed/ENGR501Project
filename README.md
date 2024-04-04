instructions:
load modules:
$module load openmpi/5.0.2 smartRedis/1.3.10 cuda/12.3

Create cluster (only needs to be done once):
redis-cli --cluster create 127.0.0.1:6378 127.0.0.1:6379 127.0.0.1:6380 
Add this flag at the end for replicas: --cluster-replicas 1

Check that cluster was succesfully allocated:
$cluster slots

Launch each server port in seperate terminals:
$redis-server ./c6378_redis.conf
$redis-server ./c6379_redis.conf
$redis-server ./py_redis.conf

create the executable:
$make all
$python launch_program.py

Shutdown the server when finished:
$redis-cli shutdown nosave

Check that it has been shut down:
$sudo lsof -i :6379
$redis-cli ping

If shutdown was unsuccesful:
$/etc/init.d/redis-server stop
