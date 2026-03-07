# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
#
# RPM spec for vdr-vaapivideo
# Build directly from tarball: rpmbuild -ta vdr-vaapivideo-<version>.tar.gz

%global pname   vaapivideo
%global __provides_exclude_from ^%{vdr_plugindir}/.*\\.so.*$

Name:           vdr-%{pname}
Version:        1.0.0
Release:        %autorelease
Summary:        VAAPI video plugin for VDR

License:        AGPL-3.0-or-later
URL:            https://github.com/dnehring7/vdr-plugin-%{pname}
Source0:        %url/archive/refs/tags/V%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  pkgconfig(alsa)
BuildRequires:  pkgconfig(libavcodec) >= 61
BuildRequires:  pkgconfig(libavfilter)
BuildRequires:  pkgconfig(libavformat)
BuildRequires:  pkgconfig(libavutil)
BuildRequires:  pkgconfig(libdrm) >= 2.4.118
BuildRequires:  pkgconfig(libswresample)
BuildRequires:  pkgconfig(libva) >= 1.22
BuildRequires:  vdr-devel >= 2.7.9
Requires:       vdr(abi)%{?_isa} = %{vdr_apiversion}

%description
Hardware-accelerated video output plugin for VDR using VAAPI decode, DRM
atomic modesetting, and ALSA audio output.

Unlike older VDR output plugins that rely on X11 or OpenGL, this plugin drives
the display directly through the kernel DRM/KMS subsystem. No display server
is required -- it runs on a bare console, in a systemd service, or headless.

%prep
%autosetup -n vdr-plugin-%{pname}-%{version}

%build
%make_build

%install
%make_install
install -dm 755 %{buildroot}%{vdr_rundir}/%{pname}
install -Dpm 644 %{name}.conf \
  %{buildroot}%{_sysconfdir}/sysconfig/vdr-plugins.d/%{pname}.conf

%files
%license LICENSE
%doc README.md
%config(noreplace) %{_sysconfdir}/sysconfig/vdr-plugins.d/%{pname}.conf
%{vdr_plugindir}/libvdr-%{pname}.so.%{vdr_apiversion}
%attr(-,%{vdr_user},root) %dir %{vdr_rundir}/%{pname}/

%changelog
%autochangelog
