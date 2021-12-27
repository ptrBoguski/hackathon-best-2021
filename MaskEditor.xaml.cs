using System;
using System.Windows;
using System.Windows.Controls;

namespace HackathonBEST
{
    public partial class MaskEditor : Window
    {
        private int[] xMask;
        private int[] yMask;
        private Action<int[]> xMaskSetter;
        private Action<int[]> yMaskSetter;
        private bool didInitialize = false;
        public MaskEditor(int[] xMask, int[]yMask, Action<int[]> xMaskSetter, Action<int[]> yMaskSetter)
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
            
            Int32.TryParse(GX1TextBox.Text, out xMask[0]);
            Int32.TryParse(GX2TextBox.Text, out xMask[1]);
            Int32.TryParse(GX3TextBox.Text, out xMask[2]);
            Int32.TryParse(GX4TextBox.Text, out xMask[3]);
            Int32.TryParse(GX5TextBox.Text, out xMask[4]);
            Int32.TryParse(GX6TextBox.Text, out xMask[5]);
            Int32.TryParse(GX7TextBox.Text, out xMask[6]);
            Int32.TryParse(GX8TextBox.Text, out xMask[7]);
            Int32.TryParse(GX9TextBox.Text, out xMask[8]);
            
            Int32.TryParse(GY1TextBox.Text, out yMask[0]);
            Int32.TryParse(GY2TextBox.Text, out yMask[1]);
            Int32.TryParse(GY3TextBox.Text, out yMask[2]);
            Int32.TryParse(GY4TextBox.Text, out yMask[3]);
            Int32.TryParse(GY5TextBox.Text, out yMask[4]);
            Int32.TryParse(GY6TextBox.Text, out yMask[5]);
            Int32.TryParse(GY7TextBox.Text, out yMask[6]);
            Int32.TryParse(GY8TextBox.Text, out yMask[7]);
            Int32.TryParse(GY9TextBox.Text, out yMask[8]);

            
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
    }
}