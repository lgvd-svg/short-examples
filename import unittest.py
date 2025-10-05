import unittest
from test_calculadora import Calculadora

class TestCalculadora(unittest.TestCase):
    
    def test_sumar(self):
        self.assertEqual(Calculadora.sumar(self,2,3),5)
        self.assertEqual(Calculadora.sumar(self,1,-1),0)

    def test_restar(self):
        self.assertEqual(Calculadora.restar(self,2,3),-1)
        self.assertEqual(Calculadora.restar(self,1,-1),2)
    
    
    def test_multiplicar(self):
        self.assertEqual(Calculadora.multiplicar(self,2,3),6)
        self.assertEqual(Calculadora.multiplicar(self,1,-1),-1)
    
    
    def test_dividir(self):
        self.assertEqual(Calculadora.dividir(self,6,3),2)
        self.assertEqual(Calculadora.dividir(self,4,1),4)
    
if __name__ == '__main__':
    unittest.main()    