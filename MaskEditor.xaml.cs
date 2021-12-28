using System;
using System.Windows;
using System.Windows.Controls;

namespace HackathonBEST
{
    public partial class MaskEditor : Window
    {
        private float[] xMask;
        private float[] yMask;
        private Action<float[]> xMaskSetter;
        private Action<float[]> yMaskSetter;
        private bool didInitialize = false;
        public MaskEditor(float[] xMask, float[]yMask, Action<float[]> xMaskSetter, Action<float[]> yMaskSetter)
        {
            InitializeComponent();
            this.xMask = xMask;
            this.yMask = yMask;
            this.xMaskSetter = xMaskSetter;
            this.yMaskSetter = yMaskSetter;
            DisplayValuesFromMask();
        }
        
        private void SaveButton_OnClick(object sender, RoutedEventArgs e)
        {
            xMaskSetter.Invoke(xMask);
            yMaskSetter.Invoke(yMask);
            Close();
        }

        public void UpdateValuesFromTextBoxes(object sender, TextChangedEventArgs e)
        {
            if (!didInitialize) return;
            
            float.TryParse(GX1TextBox.Text, out xMask[0]);
            float.TryParse(GX2TextBox.Text, out xMask[1]);
            float.TryParse(GX3TextBox.Text, out xMask[2]);
            float.TryParse(GX4TextBox.Text, out xMask[3]);
            float.TryParse(GX5TextBox.Text, out xMask[4]);
            float.TryParse(GX6TextBox.Text, out xMask[5]);
            float.TryParse(GX7TextBox.Text, out xMask[6]);
            float.TryParse(GX8TextBox.Text, out xMask[7]);
            float.TryParse(GX9TextBox.Text, out xMask[8]);
            
            float.TryParse(GY1TextBox.Text, out yMask[0]);
            float.TryParse(GY2TextBox.Text, out yMask[1]);
            float.TryParse(GY3TextBox.Text, out yMask[2]);
            float.TryParse(GY4TextBox.Text, out yMask[3]);
            float.TryParse(GY5TextBox.Text, out yMask[4]);
            float.TryParse(GY6TextBox.Text, out yMask[5]);
            float.TryParse(GY7TextBox.Text, out yMask[6]);
            float.TryParse(GY8TextBox.Text, out yMask[7]);
            float.TryParse(GY9TextBox.Text, out yMask[8]);

            
        }
        
        private void DisplayValuesFromMask()
        {
            GX1TextBox.Text = xMask[0].ToString();
            GX2TextBox.Text = xMask[1].ToString();
            GX3TextBox.Text = xMask[2].ToString();
            GX4TextBox.Text = xMask[3].ToString();
            GX5TextBox.Text = xMask[4].ToString();
            GX6TextBox.Text = xMask[5].ToString();
            GX7TextBox.Text = xMask[6].ToString();
            GX8TextBox.Text = xMask[7].ToString();
            GX9TextBox.Text = xMask[8].ToString();
            
            GY1TextBox.Text = yMask[0].ToString();
            GY2TextBox.Text = yMask[1].ToString();
            GY3TextBox.Text = yMask[2].ToString();
            GY4TextBox.Text = yMask[3].ToString();
            GY5TextBox.Text = yMask[4].ToString();
            GY6TextBox.Text = yMask[5].ToString();
            GY7TextBox.Text = yMask[6].ToString();
            GY8TextBox.Text = yMask[7].ToString();
            GY9TextBox.Text = yMask[8].ToString();
            
            didInitialize = true;
        }

        private void LoadDefaultsButton_OnClick(object sender, RoutedEventArgs e)
        {
            didInitialize = false;
            
            xMask = new float[]{1,1,1,0,0,0,-1,-1,-1};
            yMask = new float[]{1,0,-1,1,0,-1,1,0,-1};
            
            DisplayValuesFromMask();
        }

        private void CancelButton_OnClick(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}