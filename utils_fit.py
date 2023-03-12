import torch
from tqdm import tqdm

from .utils import show_result, get_lr
from .utils_metrics import PSNR, SSIM

# 定义一个epoch
def fit_one_epoch(G_model_train, D_model_train, G_model, D_model, VGG_feature_model, G_optimizer, D_optimizer, BCE_loss, MSE_loss, epoch, epoch_size, gen, Epoch, cuda, batch_size, save_interval):
    # 初始化生成器损失，鉴别器损失，PSNR，SSIM
    G_total_loss = 0
    D_total_loss = 0
    G_total_PSNR = 0
    G_total_SSIM = 0
    # 添加进度条
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            # 无需计算反向传播梯度
            with torch.no_grad():
                lr_images, hr_images    = batch
                lr_images, hr_images    = torch.from_numpy(lr_images).type(torch.FloatTensor), torch.from_numpy(hr_images).type(torch.FloatTensor)
                # torch.zeros用来将tensor中元素值全置为0，torch.ones用来将tensor中元素值全置为1
                y_real, y_fake          = torch.ones(batch_size), torch.zeros(batch_size)
                # 将数据载入GPU
                if cuda:
                    lr_images, hr_images, y_real, y_fake  = lr_images.cuda(), hr_images.cuda(), y_real.cuda(), y_fake.cuda()
            
            #-------------------------------------------------#
            #   训练判别器
            #-------------------------------------------------#

            # 鉴别器梯度归零
            D_optimizer.zero_grad()
            # 将真实图片（HR）传入鉴别器
            D_result                = D_model_train(hr_images)
            # 计算真实损失
            D_real_loss             = BCE_loss(D_result, y_real)
            # 反向传播
            D_real_loss.backward()
            # 将LR图片传入生成器得到SR
            G_result                = G_model_train(lr_images)
            # 将虚假图片（SR）传入鉴别器
            D_result                = D_model_train(G_result).squeeze()
            # 计算损失
            D_fake_loss             = BCE_loss(D_result, y_fake)
            D_fake_loss.backward()
            # 更新权重
            D_optimizer.step()

            # 对真实损失，虚假损失进行求和，得到最终鉴别器的损失
            D_train_loss            = D_real_loss + D_fake_loss

            #-------------------------------------------------#
            #   训练生成器
            #-------------------------------------------------#
            # 生成器梯度归零
            G_optimizer.zero_grad()
            # 将LR图片传入生成器得到SR
            G_result                = G_model_train(lr_images)
            # 计算SR与HR之间的图片像素损失
            image_loss              = MSE_loss(G_result, hr_images)
            # 将虚假图片（SR）传入鉴别器
            D_result                = D_model_train(G_result).squeeze()
            # 对抗损失
            adversarial_loss        = BCE_loss(D_result, y_real)
            # 内容损失
            """
            作者定义了以预训练19层VGG网络的ReLU激活层为基础的VGG loss,求生成图像和参考图像特征表示的欧氏距离。
            在已经训练好的vgg上提出某一层的feature map，将生成的图像的这一个feature map和真实图像这一个map比较。
            """
            perception_loss         = MSE_loss(VGG_feature_model(G_result), VGG_feature_model(hr_images))
            # 对像素损失、对抗损失、内容损失进行加权，得到最终生成器的损失
            G_train_loss            = image_loss + 1e-3 * adversarial_loss + 2e-6 * perception_loss 
            # 反向传播
            G_train_loss.backward()
            # 更新权重
            G_optimizer.step()
            
            G_total_loss            += G_train_loss.item()
            D_total_loss            += D_train_loss.item()
            # 无需计算反向传播梯度
            with torch.no_grad():
                # 计算PSNR
                G_total_PSNR        += PSNR(G_result, hr_images).item()
                # 计算SSIM
                G_total_SSIM        += SSIM(G_result, hr_images).item()
            # 通过set_postfix设置进度条右边的显示信息
            pbar.set_postfix(**{'G_loss'    : G_total_loss / (iteration + 1), 
                                'D_loss'    : D_total_loss / (iteration + 1), 
                                'G_PSNR'    : G_total_PSNR / (iteration + 1), 
                                'G_SSIM'    : G_total_SSIM / (iteration + 1), 
                                'lr'        : get_lr(G_optimizer)})
            pbar.update(1)
            # 每多少步保存一次结果
            if iteration % save_interval == 0:
                show_result(epoch + 1, G_model_train, lr_images, hr_images)
    # Init_Epoch为起始世代
    # Epoch总训练世代
    # epoch为当前世代
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss / epoch_size, D_total_loss / epoch_size))
    print('Saving state, iter:', str(epoch+1))
    # 每10个epoch保存一次模型权值文件（包括生成器权值和鉴别器权值）到指定路径
    if (epoch + 1) % 10==0:
        torch.save(G_model.state_dict(), 'logs/G_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%((epoch + 1), G_total_loss / epoch_size, D_total_loss / epoch_size))
        torch.save(D_model.state_dict(), 'logs/D_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%((epoch + 1), G_total_loss / epoch_size, D_total_loss / epoch_size))
