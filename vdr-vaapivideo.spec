# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
#
# RPM spec for vdr-vaapivideo
# Build directly from tarball: rpmbuild -ta vdr-vaapivideo-<version>.tar.gz

%global pname   vaapivideo
%global __provides_exclude_from ^%{vdr_libdir}/.*\\.so.*$

Name:           vdr-%{pname}
Version:        1.8.2
Release:        1%{?dist}
Summary:        VAAPI video plugin for VDR

License:        AGPL-3.0-or-later
URL:            https://github.com/dnehring7/vdr-plugin-%{pname}
Source0:        %{url}/archive/refs/tags/V%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  pkgconfig(alsa)
BuildRequires:  pkgconfig(libavcodec) >= 61
BuildRequires:  pkgconfig(libavfilter)
BuildRequires:  pkgconfig(libavformat)
BuildRequires:  pkgconfig(libavutil)
BuildRequires:  pkgconfig(libdrm)
BuildRequires:  pkgconfig(libswresample)
BuildRequires:  pkgconfig(libva-drm) >= 1.22
BuildRequires:  vdr-devel >= 2.6.6
Requires:       vdr(abi)%{?_isa} = %{vdr_apiversion}

%description
Hardware-accelerated video output plugin for VDR using VAAPI decode, DRM
atomic modesetting, and ALSA audio.

This plugin drives the display directly through the kernel DRM/KMS subsystem --
no X11, Wayland, or OpenGL required. It runs on a bare console, as a systemd
service, or fully headless.

%prep
%autosetup -n vdr-plugin-%{pname}-%{version}

%build
%make_build all probe

%install
%make_install
install -Dpm 755 vaapivideo-probe %{buildroot}%{_bindir}/vaapivideo-probe
install -Dpm 644 %{name}.conf \
  %{buildroot}%{_sysconfdir}/sysconfig/vdr-plugins.d/%{pname}.conf

%check
nm -D --defined-only %{buildroot}%{vdr_libdir}/libvdr-%{pname}.so.%{vdr_apiversion} | grep -q ' VDRPluginCreator$'
./vaapivideo-probe --help

%files
%license LICENSE
%doc README.md
%{_bindir}/vaapivideo-probe
%config(noreplace) %{_sysconfdir}/sysconfig/vdr-plugins.d/%{pname}.conf
%{vdr_libdir}/libvdr-%{pname}.so.%{vdr_apiversion}

%changelog
