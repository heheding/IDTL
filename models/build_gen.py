from models import gap
from models import modules
from models import functions


def Generator(gen):
    if gen == 'gru':
        return gap.Feature()
    if gen == 'cnn1d':
        return gap.CNN()
    if gen == 'cnn':
        return gap.CNN_SL_bn()
    if gen == 'cnnsda':
        return gap.SDACNN()
    if gen == 'lstm':
        return gap.BiLSTM()
    if gen == 'cnnlstm':
        return gap.CNNLSTM()
    if gen == 'mul':
        return gap.FeatureNet()
    if gen == 'spf':
        return gap.SPF()
    if gen == 'sar':
        return gap.SAR()
    
    return gap.Feature()


def Predictor(source):
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    if source == 'mul':
        return gap.PredNet()
    if source == 'cnn':
        return gap.Predcnn()
    
    return gap.Predictor()

def Discriminator(discri):
    if discri == 'AR':
        return gap.Discriminator_AR()
    elif discri == 'TRANs':
        return gap.Discriminator_ATT()
    elif discri == 'ATT':
        return gap.Att(dim=128)
    elif discri == 'mul':
        return gap.DiscNet()
    elif discri == 'cnn':
        return gap.DCNN()
    else:
        return gap.Domain_classifier()

def VDI(gen):
    if gen == 'UNet':
        return modules.UNet()
    if gen == 'UConcenNet':
        return modules.UConcenNet()
    if gen == 'Q_ZNet_beta':
        return modules.Q_ZNet_beta()
    if gen == 'Q_ZNet':
        return modules.Q_ZNet()
    if gen == 'Q_ZNetban1':
        return modules.Q_ZNetban1()
    if gen == 'PredNet':
        return modules.PredNet()
    if gen == 'ReconstructNet':
        return modules.ReconstructNet()
    if gen == 'ZReconstructNet':
        return modules.ZReconstructNet()
    if gen == 'DReconstructNet':
        return modules.DReconstructNet()
    if gen == 'UConcenNet':
        return modules.UConcenNet()
    if gen == 'BetaNet':
        return modules.BetaNet()
    if gen == 'Beta2UNet':
        return modules.Beta2UNet()
    if gen == 'ClassDiscNet':
        return modules.ClassDiscNet()
    if gen == 'DiscNet':
        return modules.DiscNet()
    if gen == 'GraphDNet':
        return modules.GraphDNet()
    if gen == 'sar':
        return modules.SAR()
    if gen == 'uinv':
        return modules.Uinv()
    if gen == 'uspf':
        return modules.Uspf()
    if gen == 'UReconstructNet':
        return modules.UReconstructNet()
    
    return gap.Feature()