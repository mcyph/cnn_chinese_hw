from speedysvc.client_server.shared_memory.SHMClient import SHMClient
from speedysvc.client_server.base_classes.ClientMethodsBase import ClientMethodsBase

from cnn_chinese_hw.client_server.HWServer import HWServer as srv
from cnn_chinese_hw.client_server.Singleton import Singleton


class HWClient(ClientMethodsBase,
               Singleton
               ):
    def __init__(self, client_provider=None):
        if client_provider is None:
            client_provider = SHMClient(srv)
        ClientMethodsBase.__init__(self, client_provider)

    def get_cn_written_cand(self, LStrokes, id):
        return self.send(srv.get_cn_written_cand, [
            LStrokes, id
        ])


if __name__ == '__main__':
    from pprint import pprint

    inst = HWClient()
    pprint(inst.get_cn_written_cand([[[0, 5], [5, 10], [0, 5], [5, 10]]], 500))
