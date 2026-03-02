[System.Reflection.Assembly]::LoadWithPartialName('System.Drawing') | Out-Null
[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null

$img = [System.Windows.Forms.Clipboard]::GetImage()
if($img) {
    $img.Save('C:\OSINTCOM\screenshot.png')
    Write-Host "Screenshot saved successfully"
} else {
    Write-Host "No image found in clipboard"
}
